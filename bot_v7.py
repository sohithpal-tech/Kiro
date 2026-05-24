"""
Gold Bot v7 — Four-Module Strategy System
Built on v6 foundation. Four independent modules gated by regime detector.

v7 Modules:
  MODULE 01 — TREND (SMC + Order Block + Fibonacci OTE)
  MODULE 02 — RANGE (Range Boundary SMC)
  MODULE 03 — VOLATILE SQUEEZE (Bollinger Squeeze Breakout)
  MODULE 04 — VOLATILE POST-NEWS (Post-News ICT Entry)

v6 foundations preserved: Wilder ADX, 4H bias, equity lot sizing,
state persistence, regime recovery, slippage tracking, walk-forward ML,
tiered exits, news cache, signal cooldown, session labels.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests, logging, csv, os, time, joblib, atexit, threading, subprocess, sys, json
from datetime import datetime, date, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("goldbot.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN_HERE"
CHAT_ID        = "YOUR_CHAT_ID_HERE"
MT5_LOGIN      = 12345678
MT5_PASSWORD   = "yourpassword"
MT5_SERVER     = "Exness-MT5Trial"
FINNHUB_KEY    = "YOUR_FINNHUB_KEY_HERE"

SYMBOL               = "XAUUSD"
LOG_FILE             = "trades_log.csv"
REGIME_FILE          = "regime_stats.json"
STATE_FILE           = "open_trade_state.json"
ML_MODEL_XGB         = "ml_xgb.pkl"
ML_MODEL_RF          = "ml_rf.pkl"
ML_MODEL_LR          = "ml_lr.pkl"
ATR_PERIOD           = 14
ATR_DEFAULT          = 2.0
SESSION_START_UTC    = 6
SESSION_END_UTC      = 18
NEWS_BLOCK_MINUTES   = 30
SIGNAL_INTERVAL_SEC  = 900
AUTO_LOG_INTERVAL    = 300
AUTO_RETRAIN_DAY     = 6
AUTO_RETRAIN_HOUR    = 2
MAX_RETRIES          = 3
RETRY_DELAY_SEC      = 5
SIGNAL_COOLDOWN_MINUTES = 30

_news_cache = {"result": True, "fetched_at": None}
NEWS_CACHE_MAX_AGE_MINUTES = 90

# v7 NEW SETTINGS
ADX_TREND_THRESHOLD       = 28
ADX_SILVER_BULLET_RELAXED = 24
SWEEP_BODY_FILTER         = 0.38
SESSION_END_UTC_V7        = 16
SILVER_BULLET_WINDOWS     = [(10, 11), (14, 15)]
RANGE_ADX_MAX             = 20
RANGE_MIN_CANDLES         = 8
RANGE_MIN_WIDTH_PTS       = 60
RANGE_MAX_WIDTH_PTS       = 200
RANGE_BB_STD              = 2.5
RANGE_BB_PERIOD           = 20
RANGE_REJECTION_BODY      = 0.40
SQUEEZE_ATR_LOOKBACK      = 20
SQUEEZE_BODY_MIN          = 0.45
SQUEEZE_ATR_MULT          = 2.0
POST_NEWS_WAIT_MINUTES    = 60
POST_NEWS_ATR_SPIKE_MULT  = 1.5
POST_NEWS_CONSOLIDATION   = 4
POST_NEWS_CONSOL_PCT      = 0.35
POST_NEWS_RR              = 1.5

post_news_traded_this_session = False

TRADING_MODE = "NORMAL"

MODE_SETTINGS = {
    "CONSERVATIVE": {"risk_pct": 0.5, "ml_threshold": 0.72, "max_trades": 3, "max_loss_pct": 2.0, "range_allowed": True},
    "NORMAL":       {"risk_pct": 1.0, "ml_threshold": 0.65, "max_trades": 5, "max_loss_pct": 3.0, "range_allowed": True},
    "AGGRESSIVE":   {"risk_pct": 1.5, "ml_threshold": 0.58, "max_trades": 8, "max_loss_pct": 4.0, "range_allowed": True},
}
def get_mode(k): return MODE_SETTINGS.get(TRADING_MODE, MODE_SETTINGS["NORMAL"])[k]

SPREAD_LIMITS = {
    "TREND_BULL": 30, "TREND_BEAR": 30,
    "RANGE": 20, "VOLATILE_SQUEEZE": 18, "VOLATILE_NEWS": 20, "UNKNOWN": 25,
}

trades_today = 0; last_reset = date.today(); mt5_connected = False
ml_xgb = ml_rf = ml_lr = None; TF_ENTRY = TF_TREND = TF_HTF = None
open_trade_state = {}; regime_stats = {}; last_signal_time = None

# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════
def telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e: log.error(f"Telegram: {e}")

atexit.register(lambda: telegram("⚠️ Gold Bot v7 stopped. Check your laptop/VPS."))

# ═══════════════════════════════════════════════════════════
# MT5 CONNECTION
# ═══════════════════════════════════════════════════════════
def connect_mt5(retries=MAX_RETRIES):
    global mt5_connected, TF_ENTRY, TF_TREND, TF_HTF
    for attempt in range(1, retries + 1):
        try:
            if mt5_connected and mt5.account_info() is not None: return True
            mt5.shutdown()
            if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                raise ConnectionError(f"MT5 init: {mt5.last_error()}")
            TF_ENTRY = mt5.TIMEFRAME_M15
            TF_TREND = mt5.TIMEFRAME_H1
            TF_HTF   = mt5.TIMEFRAME_H4
            if not mt5.symbol_select(SYMBOL, True):
                raise RuntimeError(f"Symbol {SYMBOL} not found")
            mt5_connected = True
            log.info("MT5 connected ✅")
            return True
        except Exception as e:
            log.error(f"MT5 attempt {attempt}: {e}")
            mt5_connected = False
            if attempt < retries: time.sleep(RETRY_DELAY_SEC)
    telegram("❌ MT5 connection failed after retries.")
    return False

def ensure_mt5():
    if not mt5_connected or mt5.account_info() is None: return connect_mt5()
    return True

def safe_run(fn, *args, retries=MAX_RETRIES, **kwargs):
    for attempt in range(1, retries + 1):
        try: return fn(*args, **kwargs)
        except Exception as e:
            log.error(f"[{fn.__name__}] {attempt}/{retries}: {e}")
            if attempt < retries: time.sleep(RETRY_DELAY_SEC)
    return None

# ═══════════════════════════════════════════════════════════
# INDICATORS (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=period - 1, adjust=False).mean()
    al    = loss.ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def calc_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def calc_adx(df, period=14):
    """ADX using correct Wilder RMA: ewm(com=period-1) — v6 Fix A1, unchanged."""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    up   = high.diff()
    down = -low.diff()
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = pd.Series(tr).ewm(com=period - 1, adjust=False).mean()
    di_p  = 100 * pd.Series(dm_p).ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_m  = 100 * pd.Series(dm_m).ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx    = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan))
    adx   = dx.ewm(com=period - 1, adjust=False).mean()
    return adx

def fetch_candles(timeframe, count=250):
    def _fetch():
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count)
        if rates is None or len(rates) < 50:
            raise ValueError("Not enough candle data")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    return safe_run(_fetch)

def get_live_atr():
    try:
        if TF_ENTRY is None: return ATR_DEFAULT
        df = fetch_candles(TF_ENTRY, ATR_PERIOD + 5)
        return round(calc_atr(df, ATR_PERIOD).iloc[-1], 2) if df is not None else ATR_DEFAULT
    except: return ATR_DEFAULT

# ═══════════════════════════════════════════════════════════
# STEP 3 — v7 REGIME DETECTION (full replacement of v6 detect_regime)
# ═══════════════════════════════════════════════════════════
def is_silver_bullet_window():
    """Returns True if current UTC hour falls within a Silver Bullet window."""
    current_hour = datetime.utcnow().hour
    for (start, end) in SILVER_BULLET_WINDOWS:
        if start <= current_hour < end:
            return True
    return False

def get_adx_threshold():
    """Returns relaxed ADX threshold during Silver Bullet windows, strict otherwise."""
    return ADX_SILVER_BULLET_RELAXED if is_silver_bullet_window() else ADX_TREND_THRESHOLD

def is_confirmed_range(df):
    """
    Confirms a structural range: 8+ candles bouncing, 60–200 point width,
    all closes within 10% tolerance of range boundaries.
    Returns (confirmed: bool, range_high: float, range_low: float)
    """
    lookback = RANGE_MIN_CANDLES + 2
    if len(df) < lookback:
        return False, None, None
    recent     = df.iloc[-lookback:]
    range_high = recent['high'].max()
    range_low  = recent['low'].min()
    range_width = range_high - range_low
    if range_width < RANGE_MIN_WIDTH_PTS or range_width > RANGE_MAX_WIDTH_PTS:
        return False, None, None
    tolerance = range_width * 0.10
    for _, candle in recent.iterrows():
        if candle['close'] > range_high + tolerance:
            return False, None, None
        if candle['close'] < range_low - tolerance:
            return False, None, None
    return True, range_high, range_low

def calc_bollinger(df, period=20, std_dev=2.5):
    """Returns (upper, mid, lower) Bollinger Bands."""
    mid   = df['close'].rolling(period).mean()
    std   = df['close'].rolling(period).std()
    return mid + std * std_dev, mid, mid - std * std_dev

def calc_bb_width(df, period=20, std_dev=2.5):
    """Returns Bollinger Band width series."""
    upper, mid, lower = calc_bollinger(df, period, std_dev)
    return upper - lower

def is_bb_squeeze(df):
    """
    Returns True if current BB width is at or very near its 252-bar minimum.
    Requires at least 260 bars to compute reliably.
    """
    if len(df) < 260:
        return False
    bw = calc_bb_width(df)
    return bw.iloc[-1] <= bw.iloc[-252:].min() * 1.05

def detect_regime(df_1h):
    """
    v7 REGIME DETECTOR — four possible outputs:
      TREND_BULL      — ADX >= threshold AND EMA50 > EMA200
      TREND_BEAR      — ADX >= threshold AND EMA50 < EMA200
      RANGE           — ADX < RANGE_ADX_MAX AND confirmed structural range
      VOLATILE_NEWS   — ATR spike > 1.5x 20-bar average
      VOLATILE_SQUEEZE — BB width at 3-month low AND ADX 20–28
      UNKNOWN         — none of the above

    Note: df_1h replaces the (df_15m, df_1h) signature from v6.
    The calling code in run_signal_engine passes df_1h directly.
    """
    if len(df_1h) < 50:
        return "UNKNOWN"
    try:
        adx   = calc_adx(df_1h).iloc[-1]
        atr_s = calc_atr(df_1h)
        atr   = atr_s.iloc[-1]
        atr20 = atr_s.iloc[-20:].mean() if len(df_1h) >= 20 else atr

        # TREND: ADX above threshold (Silver Bullet windows relax to 24)
        if adx >= get_adx_threshold():
            e50  = calc_ema(df_1h['close'], 50).iloc[-1]
            e200 = calc_ema(df_1h['close'], 200).iloc[-1]
            return "TREND_BULL" if e50 > e200 else "TREND_BEAR"

        # RANGE: confirmed structural range with low ADX
        if adx < RANGE_ADX_MAX:
            ok, _, _ = is_confirmed_range(df_1h)
            return "RANGE" if ok else "UNKNOWN"

        # VOLATILE_NEWS: ATR spike above 1.5× 20-bar average
        if atr > atr20 * POST_NEWS_ATR_SPIKE_MULT:
            return "VOLATILE_NEWS"

        # VOLATILE_SQUEEZE: BB width at 3-month low (ADX 20–28 zone)
        if is_bb_squeeze(df_1h):
            return "VOLATILE_SQUEEZE"

        return "UNKNOWN"
    except Exception as e:
        log.warning(f"detect_regime: {e}")
        return "UNKNOWN"

# ═══════════════════════════════════════════════════════════
# 4H BIAS FILTER (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def get_4h_bias(df_4h):
    """Returns BUY, SELL or None based on 4H EMA50 vs EMA200 direction."""
    df = df_4h.copy()
    df['ema50']  = calc_ema(df['close'], 50)
    df['ema200'] = calc_ema(df['close'], 200)
    last = df.iloc[-1]
    if last['ema50'] > last['ema200']: return "BUY"
    elif last['ema50'] < last['ema200']: return "SELL"
    return None

# ═══════════════════════════════════════════════════════════
# REGIME STATS — SELF-LEARNING with v6 A5 recovery (unchanged)
# ═══════════════════════════════════════════════════════════
def load_regime_stats():
    global regime_stats
    try:
        if os.path.exists(REGIME_FILE):
            with open(REGIME_FILE) as f: regime_stats = json.load(f)
    except Exception as e: log.warning(f"load_regime_stats: {e}"); regime_stats = {}

def save_regime_stats():
    try:
        with open(REGIME_FILE, 'w') as f: json.dump(regime_stats, f, indent=2)
    except Exception as e: log.error(f"save_regime_stats: {e}")

def update_regime_stats(regime, won):
    if regime not in regime_stats:
        regime_stats[regime] = {"wins": 0, "total": 0}
    regime_stats[regime]["total"] += 1
    if won: regime_stats[regime]["wins"] += 1
    save_regime_stats()

def is_regime_allowed(regime):
    """v6 A5 FIX: 14-day cooldown + 10 trial trades before permanent block. Unchanged."""
    stats       = regime_stats.get(regime, {})
    total       = stats.get("total", 0)
    if total < 30: return True
    win_rate    = stats.get("wins", 0) / total
    blocked_since = stats.get("blocked_since")
    if blocked_since:
        days_blocked = (datetime.utcnow() - datetime.fromisoformat(blocked_since)).days
        trial = stats.get("trial_trades", 0)
        if days_blocked >= 14 and trial < 10:
            regime_stats[regime]["trial_trades"] = trial + 1
            log.info(f"Regime {regime} on trial ({trial+1}/10)")
            return True
        elif trial >= 10:
            regime_stats[regime].pop("blocked_since", None)
            regime_stats[regime].pop("trial_trades", None)
    if win_rate < 0.35:
        regime_stats[regime]["blocked_since"] = datetime.utcnow().isoformat()
        log.info(f"Regime {regime} blocked — win rate {round(win_rate*100,1)}%")
        telegram(f"🚫 Regime {regime} auto-blocked (win rate: {round(win_rate*100,1)}%)")
        return False
    return True

# ═══════════════════════════════════════════════════════════
# TRADE STATE PERSISTENCE (unchanged from v6 A4)
# ═══════════════════════════════════════════════════════════
def save_trade_state():
    try:
        serializable = {str(k): v for k, v in open_trade_state.items()}
        with open(STATE_FILE, "w") as f: json.dump(serializable, f, indent=2)
    except Exception as e: log.error(f"save_trade_state: {e}")

def load_trade_state():
    global open_trade_state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f: data = json.load(f)
            open_trade_state = {int(k): v for k, v in data.items()}
            log.info(f"Trade state reloaded: {len(open_trade_state)} positions")
    except Exception as e: log.warning(f"load_trade_state: {e}")

# ═══════════════════════════════════════════════════════════
# SIGNAL HELPER FUNCTIONS
# (get_trend_1h, detect_mss, detect_fvg unchanged from v6)
# detect_liquidity_sweep: STEP 2 — body filter raised to SWEEP_BODY_FILTER (0.38)
# ═══════════════════════════════════════════════════════════
def get_trend_1h(df_1h):
    """Unchanged from v6."""
    df = df_1h.copy()
    df['ema50']  = calc_ema(df['close'], 50)
    df['ema200'] = calc_ema(df['close'], 200)
    slope = df['ema50'].iloc[-1] - df['ema50'].iloc[-5]
    last  = df.iloc[-1]
    if last['ema50'] > last['ema200'] and slope > 0: return "BUY"
    elif last['ema50'] < last['ema200'] and slope < 0: return "SELL"
    return None

def detect_liquidity_sweep(df, lookback=20):
    """
    STEP 2: body/range filter raised from 0.30 to SWEEP_BODY_FILTER (0.38).
    Everything else identical to v6 B3.
    """
    if len(df) < lookback + 2: return None
    last         = df.iloc[-1]
    candle_range = last['high'] - last['low']
    body_size    = abs(last['close'] - last['open'])
    # STEP 2 CHANGE: 0.30 → SWEEP_BODY_FILTER
    if candle_range == 0 or body_size / candle_range < SWEEP_BODY_FILTER: return None
    prev_low  = df['low'].iloc[-lookback - 1:-1].min()
    prev_high = df['high'].iloc[-lookback - 1:-1].max()
    if last['low'] < prev_low and last['close'] > prev_low:   return "BUY"
    if last['high'] > prev_high and last['close'] < prev_high: return "SELL"
    return None

def detect_mss(df, sweep_direction):
    """Unchanged from v6 B4: body >= 40% of range."""
    if len(df) < 12: return False
    last         = df.iloc[-1]
    candle_range = last['high'] - last['low']
    body_size    = abs(last['close'] - last['open'])
    if candle_range == 0 or body_size / candle_range < 0.40: return False
    if sweep_direction == "BUY":  return last['close'] > df['high'].iloc[-11:-1].max()
    if sweep_direction == "SELL": return last['close'] < df['low'].iloc[-11:-1].min()
    return False

def detect_fvg(df, direction):
    """Unchanged from v6 B2: checks gap not already filled."""
    if len(df) < 4: return None
    c0, c1, c2   = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    current_price = c0['close']
    if direction == "BUY" and c2['high'] < c0['low']:
        gap_low, gap_high = c2['high'], c0['low']
        if current_price >= gap_low: return (gap_low, gap_high)
    if direction == "SELL" and c2['low'] > c0['high']:
        gap_low, gap_high = c0['high'], c2['low']
        if current_price <= gap_high: return (gap_low, gap_high)
    return None

# ═══════════════════════════════════════════════════════════
# STEP 4 — NEW v7 SIGNAL HELPER FUNCTIONS
# (added after detect_fvg, do not replace any existing function)
# ═══════════════════════════════════════════════════════════
def detect_order_block(df, direction, mss_candle_idx=-1):
    """
    MODULE 01 helper: finds the last opposing candle before the MSS —
    this is the institutional order block.
    Returns dict with ob_low, ob_high, entry_price, sl_level or None.
    """
    if len(df) < 5: return None
    current_price = df.iloc[-1]['close']
    limit = max(-len(df), -15)
    for i in range(mss_candle_idx - 1, limit - 1, -1):
        c = df.iloc[i]
        r = c['high'] - c['low']
        if r == 0: continue
        if abs(c['close'] - c['open']) / r < 0.25: continue
        if direction == "BUY" and c['close'] < c['open']:
            ob_l, ob_h = c['close'], c['open']
            if ob_l <= current_price <= ob_h * 1.001:
                return {
                    'ob_low':      ob_l,
                    'ob_high':     ob_h,
                    'entry_price': (ob_l + ob_h) / 2,
                    'sl_level':    c['low'] - 0.5,
                }
        elif direction == "SELL" and c['close'] > c['open']:
            ob_l, ob_h = c['open'], c['close']
            if ob_l * 0.999 <= current_price <= ob_h:
                return {
                    'ob_low':      ob_l,
                    'ob_high':     ob_h,
                    'entry_price': (ob_l + ob_h) / 2,
                    'sl_level':    c['high'] + 0.5,
                }
    return None

def detect_fib_ote(df, direction, mss_candle_idx=-1, tolerance_pts=0.3):
    """
    MODULE 01 helper: checks if current price is at the 61.8% Fibonacci
    OTE (Optimal Trade Entry) level of the MSS candle's range.
    Returns dict with ote_level, entry_price, sl_level or None.
    """
    if len(df) < abs(mss_candle_idx) + 1: return None
    mc  = df.iloc[mss_candle_idx]
    rng = mc['high'] - mc['low']
    if rng < 0.5: return None
    cur = df.iloc[-1]['close']
    if direction == "BUY":
        ote = mc['high'] - 0.618 * rng
        if abs(cur - ote) <= tolerance_pts:
            return {'ote_level': ote, 'entry_price': ote, 'sl_level': mc['low'] - 0.5}
    elif direction == "SELL":
        ote = mc['low'] + 0.618 * rng
        if abs(cur - ote) <= tolerance_pts:
            return {'ote_level': ote, 'entry_price': ote, 'sl_level': mc['high'] + 0.5}
    return None

def detect_rsi_divergence(df, direction, lookback_bars=20):
    """
    MODULE 02 helper: detects bullish or bearish RSI divergence.
    Returns True if divergence confirmed, False otherwise.
    """
    df = df.copy()
    df['rsi'] = calc_rsi(df['close'])
    if len(df) < lookback_bars + 2: return False
    cur_rsi = df['rsi'].iloc[-1]
    cur_px  = df['close'].iloc[-1]
    prior   = df.iloc[-lookback_bars:-2]
    if direction == "BUY":
        idx = prior['low'].idxmin()
        if cur_px <= prior.loc[idx, 'low'] and cur_rsi > prior.loc[idx, 'rsi']:
            return True
    elif direction == "SELL":
        idx = prior['high'].idxmax()
        if cur_px >= prior.loc[idx, 'high'] and cur_rsi < prior.loc[idx, 'rsi']:
            return True
    return False

# ═══════════════════════════════════════════════════════════
# STEP 5 — v7 SIGNAL ROUTING SYSTEM
# (replaces generate_signal; four module functions + router)
# ═══════════════════════════════════════════════════════════
def generate_trend_signal(df_15m, df_1h, df_4h):
    """
    MODULE 01 — TREND (SMC + Order Block + Fibonacci OTE)
    Active when regime is TREND_BULL or TREND_BEAR.
    Session: 06:00–16:00 UTC (removed 16:30–18:00 late window).
    Three entry types in priority order: FVG → OB → OTE.
    ADX threshold relaxed during Silver Bullet windows (10–11, 14–15 UTC).
    Sweep body filter raised to 38% (SWEEP_BODY_FILTER).
    """
    current_hour = datetime.utcnow().hour
    if not (SESSION_START_UTC <= current_hour < SESSION_END_UTC_V7):
        return None

    df_15m = df_15m.copy()
    df_15m['rsi'] = calc_rsi(df_15m['close'])
    rsi = df_15m['rsi'].iloc[-1]

    # 1H trend direction
    trend_dir = get_trend_1h(df_1h)
    if trend_dir is None: return None

    # 4H bias filter
    if df_4h is not None:
        b4 = get_4h_bias(df_4h)
        if b4 is not None and trend_dir != b4: return None

    # Liquidity sweep (body >= SWEEP_BODY_FILTER = 0.38)
    sweep = detect_liquidity_sweep(df_15m)
    if sweep is None or sweep != trend_dir: return None

    # RSI guard
    if sweep == "BUY"  and rsi > 65: return None
    if sweep == "SELL" and rsi < 35: return None

    # MSS confirmation (body >= 0.40)
    if not detect_mss(df_15m, sweep): return None

    atr = calc_atr(df_15m).iloc[-1]

    # Entry type 1: FVG (highest priority)
    fvg = detect_fvg(df_15m, sweep)
    if fvg:
        fl, fh = fvg
        ep = (fl + fh) / 2
        sl = (fl - atr * 0.5) if sweep == "BUY" else (fh + atr * 0.5)
        return {'direction': sweep, 'entry_type': 'FVG', 'entry_price': ep, 'sl': sl}

    # Entry type 2: Order Block
    ob = detect_order_block(df_15m, sweep)
    if ob:
        return {
            'direction':   sweep,
            'entry_type':  'OB',
            'entry_price': ob['entry_price'],
            'sl':          ob['sl_level'],
        }

    # Entry type 3: Fibonacci OTE
    ote = detect_fib_ote(df_15m, sweep, mss_candle_idx=-2)
    if ote:
        return {
            'direction':   sweep,
            'entry_type':  'OTE',
            'entry_price': ote['entry_price'],
            'sl':          ote['sl_level'],
        }

    return None


def generate_range_signal(df_15m):
    """
    MODULE 02 — RANGE (Range Boundary SMC)
    Active when regime is RANGE (ADX < 20, confirmed structural range).
    Session: 07:00–15:00 UTC (peak liquidity hours for range trading).
    Requires: range rejection candle + RSI divergence + BB extreme touch.
    Fully replaces the broken v6 RSI-only range strategy.
    """
    current_hour = datetime.utcnow().hour
    if not (7 <= current_hour < 15): return None

    ok, rh, rl = is_confirmed_range(df_15m)
    if not ok: return None

    rw  = rh - rl
    mid = rl + rw / 2

    df_15m = df_15m.copy()
    df_15m['rsi'] = calc_rsi(df_15m['close'])

    bbu, _, bbl = calc_bollinger(df_15m, RANGE_BB_PERIOD, RANGE_BB_STD)
    prev = df_15m.iloc[-2]
    pr   = prev['high'] - prev['low']
    pb   = abs(prev['close'] - prev['open'])
    if pr == 0: return None
    bpct = pb / pr
    ep   = df_15m.iloc[-1]['close']

    # SELL at range high: wick above range_high, close below, body>=40%, RSI divergence, BB touch
    if (prev['high'] > rh
            and prev['close'] < rh
            and bpct >= RANGE_REJECTION_BODY
            and detect_rsi_divergence(df_15m, "SELL")
            and prev['high'] >= bbu.iloc[-1]):
        return {
            'direction':  'SELL',
            'entry_price': ep,
            'sl':          prev['high'] + 0.5,
            'tp1':         mid,
            'tp2':         rl + 2.0,
            'range_high':  rh,
            'range_low':   rl,
        }

    # BUY at range low: wick below range_low, close above, body>=40%, RSI divergence, BB touch
    if (prev['low'] < rl
            and prev['close'] > rl
            and bpct >= RANGE_REJECTION_BODY
            and detect_rsi_divergence(df_15m, "BUY")
            and prev['low'] <= bbl.iloc[-1]):
        return {
            'direction':  'BUY',
            'entry_price': ep,
            'sl':          prev['low'] - 0.5,
            'tp1':         mid,
            'tp2':         rh - 2.0,
            'range_high':  rh,
            'range_low':   rl,
        }

    return None


def generate_squeeze_signal(df_15m, df_4h):
    """
    MODULE 03 — VOLATILE SQUEEZE (Bollinger Squeeze Breakout)
    Active when regime is VOLATILE_SQUEEZE:
      ATR at 20-period low AND BB width at 3-month low AND ADX 20–28.
    Session: 07:00–16:00 UTC.
    Avoids 2-hour window around high-impact news events.
    Requires: ATR at minimum + BB breakout candle with body >= 45% + 4H alignment.
    """
    current_hour = datetime.utcnow().hour
    if not (7 <= current_hour < 16): return None

    # Block if within 2 hours of a high-impact event
    try:
        now = datetime.utcnow()
        url = (f"https://finnhub.io/api/v1/calendar/economic"
               f"?from={now.strftime('%Y-%m-%d')}&to={now.strftime('%Y-%m-%d')}"
               f"&token={FINNHUB_KEY}")
        events = requests.get(url, timeout=5).json().get('economicCalendar', [])
        for ev in events:
            if ev.get('impact', '') == 'high':
                et = datetime.strptime(ev['time'], "%Y-%m-%d %H:%M:%S")
                if abs((now - et).total_seconds() / 60) < 120:
                    return None
    except Exception:
        return None  # if news API fails, skip squeeze entry as precaution

    b4 = get_4h_bias(df_4h) if df_4h is not None else None

    df_15m  = df_15m.copy()
    atr_s   = calc_atr(df_15m)
    atr_now = atr_s.iloc[-1]

    # ATR must be at or near its 20-bar minimum
    if atr_now > atr_s.iloc[-SQUEEZE_ATR_LOOKBACK:].min() * 1.08:
        return None

    atr_pre = atr_now
    bbu, _, bbl = calc_bollinger(df_15m, RANGE_BB_PERIOD, RANGE_BB_STD)
    last = df_15m.iloc[-1]
    cr   = last['high'] - last['low']
    cb   = abs(last['close'] - last['open'])

    # Breakout candle must have body >= 45%
    if cr == 0 or cb / cr < SQUEEZE_BODY_MIN: return None

    # Determine breakout direction from BB extreme
    d = None
    if last['close'] > bbu.iloc[-1] and last['close'] > last['open']:
        d = "BUY"
    elif last['close'] < bbl.iloc[-1] and last['close'] < last['open']:
        d = "SELL"
    if d is None: return None

    # 4H bias alignment
    if b4 is not None and d != b4: return None

    ep = last['close']
    td = atr_pre * SQUEEZE_ATR_MULT
    sl = (last['low']  - 0.3) if d == "BUY"  else (last['high'] + 0.3)
    tp = (ep + td)             if d == "BUY"  else (ep - td)

    return {'direction': d, 'entry_price': ep, 'sl': sl, 'tp': tp}


def generate_post_news_signal(df_15m):
    """
    MODULE 04 — VOLATILE POST-NEWS (Post-News ICT Entry)
    Active when regime is VOLATILE_NEWS.
    Waits 60 minutes after an ATR spike, then looks for price consolidation
    followed by a sweep + FVG entry in the spike direction.
    One trade per session maximum (post_news_traded_this_session guard).
    """
    global post_news_traded_this_session
    if post_news_traded_this_session: return None

    current_hour = datetime.utcnow().hour
    if not (7 <= current_hour < 16): return None

    atr_s = calc_atr(df_15m)
    avg   = atr_s.iloc[-20:].mean()

    # Find ATR spike candle in the last 6 bars
    spike = None
    for i in range(-6, -1):
        c = df_15m.iloc[i]
        if (c['high'] - c['low']) > avg * POST_NEWS_ATR_SPIKE_MULT:
            spike = {
                'direction':   "BUY" if c['close'] > c['open'] else "SELL",
                'spike_time':  c.get('time', datetime.utcnow()),
                'spike_range': c['high'] - c['low'],
            }
            break
    if spike is None: return None

    # Normalise spike time to naive datetime
    st = spike['spike_time']
    if hasattr(st, 'to_pydatetime'):
        st = st.to_pydatetime().replace(tzinfo=None)

    # Must wait POST_NEWS_WAIT_MINUTES (60 min) after spike
    if (datetime.utcnow() - st).total_seconds() / 60 < POST_NEWS_WAIT_MINUTES:
        return None

    # Consolidation check: last 4 candles must be within 35% of spike range
    recent = df_15m.iloc[-POST_NEWS_CONSOLIDATION:]
    if recent['high'].max() - recent['low'].min() > spike['spike_range'] * POST_NEWS_CONSOL_PCT:
        return None

    d = spike['direction']

    # Sweep in spike direction
    if detect_liquidity_sweep(df_15m) != d: return None

    # FVG entry confirmation
    fvg = detect_fvg(df_15m, d)
    if fvg is None: return None

    fl, fh = fvg
    ep     = (fl + fh) / 2
    av     = atr_s.iloc[-1]
    sl     = (fl - av * 0.5) if d == "BUY" else (fh + av * 0.5)
    sd     = abs(ep - sl)
    tp     = (ep + sd * POST_NEWS_RR) if d == "BUY" else (ep - sd * POST_NEWS_RR)

    post_news_traded_this_session = True

    return {'direction': d, 'entry_price': ep, 'sl': sl, 'tp': tp}


def generate_signal(df_15m, df_1h, df_4h, regime):
    """
    STEP 5 — v7 SIGNAL ROUTER
    Routes to the correct module based on regime. Returns direction string or None.
    Only one module fires at a time.
    """
    if regime in ("TREND_BULL", "TREND_BEAR"):
        r = generate_trend_signal(df_15m, df_1h, df_4h)
        return r['direction'] if r else None

    elif regime == "RANGE":
        r = generate_range_signal(df_15m)
        return r['direction'] if r else None

    elif regime == "VOLATILE_SQUEEZE":
        r = generate_squeeze_signal(df_15m, df_4h)
        return r['direction'] if r else None

    elif regime == "VOLATILE_NEWS":
        r = generate_post_news_signal(df_15m)
        return r['direction'] if r else None

    return None

# ═══════════════════════════════════════════════════════════
# ENSEMBLE ML (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def load_ml_models():
    global ml_xgb, ml_rf, ml_lr
    loaded = []
    for name, attr, file in [
        ("XGBoost",      "ml_xgb", ML_MODEL_XGB),
        ("RandomForest", "ml_rf",  ML_MODEL_RF),
        ("LogisticReg",  "ml_lr",  ML_MODEL_LR),
    ]:
        try: globals()[attr] = joblib.load(file); loaded.append(name)
        except Exception as e: log.info(f"{name} not found: {e}")
    log.info(f"ML models loaded: {loaded}" if loaded else "No ML models — placeholder mode")

def build_ml_features(signal, atr, sl_dist, regime, rsi_val=50.0):
    """v6 A2 FIX: dynamic rr_quality + rsi_at_entry. Unchanged."""
    now         = datetime.utcnow()
    hour        = now.hour
    dow         = now.weekday()
    session_enc = 0 if 6 <= hour < 12 else 1 if 12 <= hour < 18 else 2
    sig_enc     = 1 if signal == "BUY" else 0
    tp_dist     = sl_dist * 2.0
    rr          = round(tp_dist / sl_dist, 3) if sl_dist > 0 else 2.0
    sl_atr      = round(sl_dist / atr, 3) if atr > 0 else 1.0
    atr_tier    = 0 if atr < 1.5 else 1 if atr < 3.0 else 2
    regime_enc  = {
        "TREND_BULL": 0, "TREND_BEAR": 1, "RANGE": 2,
        "VOLATILE_SQUEEZE": 3, "VOLATILE_NEWS": 3, "UNKNOWN": 4,
    }.get(regime, 4)
    lot = calculate_lot(sl_dist)
    return pd.DataFrame([{
        "sl_distance":  sl_dist,
        "rr_quality":   rr,
        "atr_used":     atr,
        "sl_atr_ratio": sl_atr,
        "lot":          lot,
        "hour":         hour,
        "day_of_week":  dow,
        "session_enc":  session_enc,
        "signal_enc":   sig_enc,
        "atr_tier":     atr_tier,
        "regime_enc":   regime_enc,
        "rsi_at_entry": round(rsi_val, 2),
    }])

def ml_filter(signal, atr, sl_dist, regime, rsi_val=50.0):
    """Unchanged from v6."""
    threshold = get_mode("ml_threshold")
    if ml_xgb is None and ml_rf is None and ml_lr is None: return True
    features = build_ml_features(signal, atr, sl_dist, regime, rsi_val)
    scores, weights = [], []
    try:
        if ml_xgb: scores.append(ml_xgb.predict_proba(features)[0][1]); weights.append(0.4)
    except Exception as e: log.warning(f"XGBoost: {e}")
    try:
        if ml_rf: scores.append(ml_rf.predict_proba(features)[0][1]); weights.append(0.3)
    except Exception as e: log.warning(f"RF: {e}")
    try:
        if ml_lr: scores.append(ml_lr.predict_proba(features)[0][1]); weights.append(0.3)
    except Exception as e: log.warning(f"LR: {e}")
    if not scores: return True
    total_w = sum(weights)
    final   = sum(s * w for s, w in zip(scores, weights)) / total_w
    log.info(f"ML score: {round(final*100,1)}% (threshold: {int(threshold*100)}%)")
    if final < threshold:
        telegram(f"🤖 ML rejected | Score: {round(final*100,1)}% < {int(threshold*100)}%")
        return False
    return True

# ═══════════════════════════════════════════════════════════
# FILTERS (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def session_filter():
    return SESSION_START_UTC <= datetime.utcnow().hour < SESSION_END_UTC

def spread_filter(regime="UNKNOWN"):
    try:
        tick  = mt5.symbol_info_tick(SYMBOL)
        info  = mt5.symbol_info(SYMBOL)
        if tick is None or info is None: return False
        spread = (tick.ask - tick.bid) / info.point
        limit  = SPREAD_LIMITS.get(regime, 25)
        if spread > limit:
            log.info(f"Spread {spread:.1f}pts > {limit} for {regime}")
            return False
        return True
    except Exception as e: log.error(f"Spread: {e}"); return False

def news_filter():
    """v6 A3 FIX: cache + fail-safe. Unchanged."""
    try:
        now = datetime.utcnow()
        url = (f"https://finnhub.io/api/v1/calendar/economic"
               f"?from={now.strftime('%Y-%m-%d')}&to={now.strftime('%Y-%m-%d')}"
               f"&token={FINNHUB_KEY}")
        events   = requests.get(url, timeout=5).json().get("economicCalendar", [])
        keywords = ["nonfarm","nfp","cpi","inflation","fomc","fed","interest rate","gdp","unemployment"]
        allow_trade = True
        for ev in events:
            if ev.get("impact", "").lower() != "high": continue
            if not any(k in ev.get("event", "").lower() for k in keywords): continue
            try:
                ev_dt = datetime.strptime(ev.get("time", ""), "%Y-%m-%d %H:%M:%S")
                if ev_dt - timedelta(minutes=NEWS_BLOCK_MINUTES) <= now <= ev_dt + timedelta(minutes=NEWS_BLOCK_MINUTES):
                    msg = f"📰 News block: {ev.get('event')} @ {ev.get('time')}"
                    log.info(msg); telegram(msg)
                    allow_trade = False; break
            except Exception: continue
        _news_cache["result"]     = allow_trade
        _news_cache["fetched_at"] = datetime.utcnow()
        return allow_trade
    except Exception as e:
        log.warning(f"News filter error: {e}")
        if _news_cache["fetched_at"] is not None:
            age = (datetime.utcnow() - _news_cache["fetched_at"]).total_seconds() / 60
            if age < NEWS_CACHE_MAX_AGE_MINUTES:
                log.info(f"Using cached news result (age: {age:.0f}min)")
                return _news_cache["result"]
        log.warning("News cache stale — blocking trade as safe default")
        telegram("WARNING: News filter API down + cache stale — trade blocked")
        return False

def get_todays_loss():
    try:
        t0    = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(t0, datetime.utcnow())
        if not deals: return 0.0
        return round(sum(abs(d.profit) for d in deals
                         if d.profit < 0 and d.entry == mt5.DEAL_ENTRY_OUT), 2)
    except: return 0.0

def daily_reset():
    global trades_today, last_reset
    today = date.today()
    if today != last_reset:
        trades_today = 0; last_reset = today
        log.info("Daily reset")

def risk_filter():
    global trades_today
    daily_reset()
    if trades_today >= get_mode("max_trades"):
        telegram(f"🚫 Max trades reached [{TRADING_MODE}]"); return False
    account = mt5.account_info()
    if account is None: return False
    if get_todays_loss() >= account.balance * (get_mode("max_loss_pct") / 100):
        telegram(f"🚫 Daily loss limit hit"); return False
    return True

def has_open_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos is not None and any(p.magic == 111444 for p in pos)

def calculate_lot(sl_distance_price):
    """v6 A6 FIX: uses account.equity not account.balance. Unchanged."""
    try:
        account = mt5.account_info()
        info    = mt5.symbol_info(SYMBOL)
        if account is None or info is None or info.trade_tick_size == 0:
            return getattr(info, 'volume_min', 0.01) if info else 0.01
        risk_amount   = account.equity * (get_mode("risk_pct") / 100)
        value_per_lot = info.trade_tick_value / info.trade_tick_size
        lot           = risk_amount / (sl_distance_price * value_per_lot)
        return max(info.volume_min, min(round(lot, 2), info.volume_max))
    except: return 0.01

# ═══════════════════════════════════════════════════════════
# TRADE LOGGER (unchanged from v6 — B5 slippage tracking)
# ═══════════════════════════════════════════════════════════
def setup_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "date","time","signal","symbol","price","fill_price",
                "slippage_pts","sl","tp_ref","lot","risk_pct",
                "sl_distance","rr_ref","atr_used","account_balance",
                "regime","mode","result","pnl_usd","session","ticket",
            ])

def log_trade(signal, price, fill_price, sl, tp_ref, lot, atr, balance, ticket, regime, slippage_pts=0):
    """v6 B5: fill_price and slippage_pts logged. v6 C3: LDN_NY_Overlap label. Unchanged."""
    now     = datetime.utcnow()
    sl_dist = round(abs(price - sl), 2)
    tp_dist = round(abs(price - tp_ref), 2)
    rr      = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
    hour    = now.hour
    if 12 <= hour < 15:   session = "LDN_NY_Overlap"
    elif 6 <= hour < 12:  session = "London"
    elif 15 <= hour < 18: session = "NY"
    else:                 session = "Asian"
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
            signal, SYMBOL, price, fill_price, slippage_pts,
            sl, tp_ref, lot, get_mode("risk_pct"),
            sl_dist, rr, atr, round(balance, 2),
            regime, TRADING_MODE, "", "", session, ticket,
        ])
    log.info(f"Trade logged | Ticket:{ticket} | Regime:{regime} | Slippage:{slippage_pts}pts")

# ═══════════════════════════════════════════════════════════
# AUTO CSV UPDATER (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def auto_update_csv():
    while True:
        try:
            time.sleep(AUTO_LOG_INTERVAL)
            if not ensure_mt5() or not os.path.exists(LOG_FILE): continue
            df_csv = pd.read_csv(LOG_FILE, dtype=str)
            if 'ticket' not in df_csv.columns: continue
            from_dt = datetime.utcnow() - timedelta(days=30)
            deals   = mt5.history_deals_get(from_dt, datetime.utcnow())
            if not deals: continue
            updated = False
            for deal in deals:
                if deal.entry != mt5.DEAL_ENTRY_OUT: continue
                mask = df_csv['ticket'] == str(deal.order)
                if not mask.any(): continue
                existing = str(df_csv.loc[mask, 'result'].values[0]).strip()
                if existing and existing.lower() not in ("", "nan"): continue
                result = "WIN" if deal.profit > 0 else "LOSS"
                df_csv.loc[mask, 'result']  = result
                df_csv.loc[mask, 'pnl_usd'] = round(deal.profit, 2)
                updated = True
                regime_val = str(df_csv.loc[mask, 'regime'].values[0]).strip()
                if regime_val and regime_val not in ("nan", ""):
                    update_regime_stats(regime_val, deal.profit > 0)
                log.info(f"CSV updated: {deal.order} → {result}")
            if updated: df_csv.to_csv(LOG_FILE, index=False)
        except Exception as e: log.error(f"auto_update_csv: {e}")

# ═══════════════════════════════════════════════════════════
# AUTO RETRAIN (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def auto_retrain_scheduler():
    while True:
        try:
            now = datetime.utcnow()
            if now.weekday() == AUTO_RETRAIN_DAY and now.hour == AUTO_RETRAIN_HOUR:
                telegram("🔁 Weekly ML retrain starting...")
                for script in ["prepare_data.py", "train_model.py"]:
                    r = subprocess.run([sys.executable, script],
                                       capture_output=True, text=True, timeout=300)
                    if r.returncode != 0:
                        raise RuntimeError(f"{script}: {r.stderr[:200]}")
                load_ml_models()
                telegram("✅ ML retrained and reloaded")
                time.sleep(7200)
            else:
                time.sleep(3600)
        except Exception as e:
            log.error(f"auto_retrain: {e}")
            telegram(f"❌ ML retrain failed: {e}")
            time.sleep(3600)

# ═══════════════════════════════════════════════════════════
# TIERED PARTIAL EXIT + TRAILING SL (unchanged from v6)
# ═══════════════════════════════════════════════════════════
def manage_open_trades():
    if not ensure_mt5(): return
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return
    atr  = get_live_atr()
    info = mt5.symbol_info(SYMBOL)
    if info is None: return
    min_vol = info.volume_min
    for pos in positions:
        if pos.magic != 111444 or pos.ticket not in open_trade_state: continue
        state     = open_trade_state[pos.ticket]
        entry     = state['entry_price']
        init_sl   = state['initial_sl']
        one_r     = abs(entry - init_sl)
        direction = state['direction']
        init_size = state['initial_size']
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None: continue
        price_now = tick.bid if direction == "BUY" else tick.ask
        if not state.get('p1_done'):
            reached = ((direction == "BUY"  and price_now >= entry + one_r) or
                       (direction == "SELL" and price_now <= entry - one_r))
            if reached:
                vol = round(init_size * 0.30, 2)
                if vol >= min_vol:
                    if _partial_close(pos.ticket, vol, direction, price_now):
                        state['p1_done'] = True
                        _modify_sl(pos.ticket, entry)
                        telegram(f"✂️ 30% @+1R | Ticket:{pos.ticket} | BE set")
                        save_trade_state()
                else:
                    state['p1_done'] = True
        elif not state.get('p2_done'):
            reached = ((direction == "BUY"  and price_now >= entry + 2 * one_r) or
                       (direction == "SELL" and price_now <= entry - 2 * one_r))
            if reached:
                vol = round(init_size * 0.30, 2)
                if vol >= min_vol:
                    if _partial_close(pos.ticket, vol, direction, price_now):
                        state['p2_done'] = True
                        telegram(f"✂️ 30% @+2R | Ticket:{pos.ticket}")
                        save_trade_state()
                else:
                    state['p2_done'] = True
        elif state.get('p1_done') and state.get('p2_done'):
            atr_sl = atr * 1.5
            if direction == "BUY":
                new_sl = round(price_now - atr_sl, 2)
                if new_sl > pos.sl: _modify_sl(pos.ticket, new_sl)
            elif direction == "SELL":
                new_sl = round(price_now + atr_sl, 2)
                if new_sl < pos.sl or pos.sl == 0: _modify_sl(pos.ticket, new_sl)

def _partial_close(ticket, vol, direction, price):
    res = mt5.order_send({
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   SYMBOL,
        "volume":   vol,
        "type":     mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY,
        "price":    price,
        "position": ticket,
        "deviation": 10,
        "magic":    111444,
        "comment":  "GoldBot_Partial_v7",
    })
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"Partial close OK | Ticket:{ticket} | Vol:{vol}")
        return True
    log.error(f"Partial close failed: {res}")
    return False

def _modify_sl(ticket, new_sl):
    """Never sends TP — only modifies SL. Unchanged from v6."""
    try:
        if not mt5.positions_get(ticket=ticket): return
        res = mt5.order_send({
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   SYMBOL,
            "position": ticket,
            "sl":       new_sl,
        })
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"SL → {new_sl} | Ticket:{ticket}")
        else:
            log.error(f"SL modify failed: {res}")
    except Exception as e: log.error(f"_modify_sl: {e}")

# ═══════════════════════════════════════════════════════════
# STEP 7 — TRADE EXECUTION (comment field → GoldBot_v7_{regime})
# ═══════════════════════════════════════════════════════════
def place_trade(signal, regime):
    global trades_today
    if not ensure_mt5(): telegram("❌ MT5 not connected"); return
    if not session_filter(): return
    if has_open_position(): return
    if not spread_filter(regime): telegram(f"⚠️ Spread too high for {regime}"); return
    if not news_filter(): return
    if not risk_filter(): return
    if not is_regime_allowed(regime): return

    tick    = mt5.symbol_info_tick(SYMBOL)
    account = mt5.account_info()
    if tick is None or account is None: telegram("❌ Cannot get market data"); return

    atr = get_live_atr()

    if signal == "BUY":
        price  = tick.ask
        sl     = round(price - atr * 1.5, 2)
        tp_ref = round(price + atr * 3.0, 2)
        otype  = mt5.ORDER_TYPE_BUY
    else:
        price  = tick.bid
        sl     = round(price + atr * 1.5, 2)
        tp_ref = round(price - atr * 3.0, 2)
        otype  = mt5.ORDER_TYPE_SELL

    sl_dist = abs(price - sl)
    tp_dist = abs(price - tp_ref)
    if sl_dist == 0: telegram("❌ SL distance zero"); return
    if tp_dist / sl_dist < 2.0: telegram(f"⚠️ RR too low"); return

    rsi_now = 50.0
    try:
        df_tmp = fetch_candles(TF_ENTRY, 20)
        if df_tmp is not None:
            rsi_now = round(calc_rsi(df_tmp['close']).iloc[-1], 2)
    except: pass

    if not ml_filter(signal, atr, sl_dist, regime, rsi_now): return

    lot    = calculate_lot(sl_dist)
    info   = mt5.symbol_info(SYMBOL)

    # STEP 7: comment field → GoldBot_v7_{regime}
    result = mt5.order_send({
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   SYMBOL,
        "volume":   lot,
        "type":     otype,
        "price":    price,
        "sl":       sl,
        "deviation": 10,
        "magic":    111444,
        "comment":  f"GoldBot_v7_{regime}",   # STEP 7 change
    })
    if result is None: telegram("❌ order_send None"); return

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        trades_today += 1
        ticket      = result.order
        fill_price  = result.price
        slippage_pts = round(abs(fill_price - price) / info.point, 1) if info else 0
        open_trade_state[ticket] = {
            "p1_done": False, "p2_done": False,
            "initial_sl":   sl,
            "entry_price":  price,
            "initial_size": lot,
            "direction":    signal,
            "regime":       regime,
        }
        save_trade_state()
        log_trade(signal, price, fill_price, sl, tp_ref, lot, atr,
                  account.balance, ticket, regime, slippage_pts)
        # STEP 7: Telegram message updated to v7
        telegram(
            f"✅ {signal} EXECUTED (v7)\n"
            f"Module: {regime} | Mode: {TRADING_MODE}\n"
            f"Price: {price} | Fill: {fill_price} | Slip: {slippage_pts}pts\n"
            f"ATR: {atr} | SL: {sl} | Lot: {lot} | Risk: {get_mode('risk_pct')}%\n"
            f"Exits: 30%@1R → 30%@2R → 40% trail | Ticket: {ticket}"
        )
    else:
        err = f"❌ Order failed: {result.retcode} — {result.comment}"
        telegram(err); log.error(err)

# ═══════════════════════════════════════════════════════════
# SIGNAL ENGINE — STEP 6: post-news session reset added
# ═══════════════════════════════════════════════════════════
def run_signal_engine():
    global last_signal_time, post_news_traded_this_session
    log.info("Signal engine started — v7 four-module routing — 15 min intervals")
    while True:
        try:
            time.sleep(SIGNAL_INTERVAL_SEC)
            if not ensure_mt5(): continue

            manage_open_trades()

            if has_open_position() or not session_filter(): continue

            # v6 C2: 30-minute signal cooldown
            now = datetime.utcnow()
            if last_signal_time is not None:
                minutes_since = (now - last_signal_time).total_seconds() / 60
                if minutes_since < SIGNAL_COOLDOWN_MINUTES:
                    log.info(f"Signal cooldown active ({minutes_since:.0f}min since last)")
                    continue

            # STEP 6: post-news session guard reset at session start
            if now.hour == SESSION_START_UTC and now.minute < 15:
                if post_news_traded_this_session:
                    post_news_traded_this_session = False
                    log.info("Post-news session guard reset.")

            # Fetch all required timeframes
            df_1h  = fetch_candles(TF_TREND, 300)   # 300 bars for BB squeeze check
            df_15m = fetch_candles(TF_ENTRY, 100)
            df_4h  = fetch_candles(TF_HTF, 250)
            if df_1h is None or df_15m is None: continue

            # v7: detect_regime takes df_1h only
            regime = detect_regime(df_1h)
            log.info(f"Regime: {regime}")

            # Block unresolved regimes
            if regime == "UNKNOWN":
                log.info("Regime UNKNOWN — no new trades")
                continue

            if not spread_filter(regime): continue

            signal = generate_signal(df_15m, df_1h, df_4h, regime)
            if signal is None: continue

            if not news_filter() or not risk_filter(): continue

            log.info(f"✅ Signal: {signal} | Regime: {regime}")
            last_signal_time = now
            place_trade(signal, regime)

        except Exception as e:
            log.error(f"Signal engine error: {e}")

# ═══════════════════════════════════════════════════════════
# FLASK ENDPOINTS (unchanged from v6)
# ═══════════════════════════════════════════════════════════
@app.route('/webhook', methods=['POST'])
def webhook():
    data   = request.get_json(silent=True)
    if not data: return {"status": "error", "msg": "no json"}, 400
    signal = data.get("signal", "").upper()
    regime = data.get("regime", "UNKNOWN")
    if signal not in ("BUY", "SELL"):
        return {"status": "error", "msg": "invalid signal"}, 400
    log.info(f"Webhook: {signal}")
    place_trade(signal, regime)
    return {"status": "ok"}

@app.route('/health', methods=['GET'])
def health():
    connected = ensure_mt5()
    df_1h     = fetch_candles(TF_TREND, 300) if connected else None
    regime    = detect_regime(df_1h) if df_1h is not None else "N/A"
    return jsonify({
        "version":        "v7",
        "status":         "running",
        "mode":           TRADING_MODE,
        "mt5_connected":  connected,
        "regime":         regime,
        "trades_today":   trades_today,
        "loss_today":     f"${get_todays_loss():.2f}" if connected else "N/A",
        "live_atr":       get_live_atr() if connected else "N/A",
        "open_trades":    len(open_trade_state),
        "ml_status":      f"XGB:{'✅' if ml_xgb else '❌'} RF:{'✅' if ml_rf else '❌'} LR:{'✅' if ml_lr else '❌'}",
        "regime_stats":   regime_stats,
        "silver_bullet":  is_silver_bullet_window(),
        "post_news_guard": post_news_traded_this_session,
    })

@app.route('/regime_stats', methods=['GET'])
def get_regime_stats_endpoint():
    stats = {}
    for r, s in regime_stats.items():
        t  = s.get("total", 0)
        w  = s.get("wins",  0)
        wr = round(w / t * 100, 1) if t > 0 else 0
        stats[r] = {"total": t, "wins": w, "win_rate": f"{wr}%",
                    "allowed": is_regime_allowed(r)}
    return jsonify(stats)

# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════
def start_threads():
    for fn, name in [
        (run_signal_engine,      "SignalEngine"),
        (auto_update_csv,        "CSVUpdater"),
        (auto_retrain_scheduler, "MLRetrain"),
    ]:
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        log.info(f"Thread started: {name}")

if __name__ == "__main__":
    setup_log()
    load_regime_stats()
    load_trade_state()
    load_ml_models()
    if connect_mt5():
        telegram(
            f"🤖 Gold Bot v7 started ✅\n"
            f"Mode: {TRADING_MODE}\n"
            f"Modules: TREND (OB+OTE+FVG) | RANGE (SMC) | SQUEEZE | POST-NEWS\n"
            f"Silver Bullet: {SILVER_BULLET_WINDOWS} UTC | Session: 06:00–16:00\n"
            f"Sweep filter: {int(SWEEP_BODY_FILTER*100)}% body | ADX trend: {ADX_TREND_THRESHOLD}"
        )
    else:
        telegram("⚠️ Bot v7 started — MT5 connection failed, retrying")
    start_threads()
    app.run(host="0.0.0.0", port=5000)
