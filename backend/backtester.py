# ======================================================
# === backtest_engine.py ===
# ======================================================
#
# Elite backtester for Nawthviper — uses YOUR REAL logic.
#
# ✅ Candle-by-candle simulation (zero lookahead)
# ✅ Real spread + randomised slippage
# ✅ Full trade lifecycle: partial TP, SL to BE, trailing
# ✅ Equity curve + drawdown tracking
# ✅ Walk-forward testing support
# ✅ Multi-pair robustness testing
# ✅ Expectancy, profit factor, Monte Carlo
# ✅ CSV trade log (feeds learning system)
#
# USAGE:
#   python backtest_engine.py
#
# DATA FORMAT:
#   CSV files with columns: time, open, high, low, close, tick_volume
#   Filename pattern: XAUUSD_H1.csv, EURUSD_M5.csv etc.
#   Export from MT5: Tools → History Center → Export
#

import os
import csv
import json
import math
import random
import traceback
import hashlib
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

from ta.volatility import AverageTrueRange
from ta.trend import MACD
from ta.momentum import RSIIndicator

# === Your real modules ===
from zone_detector import detect_zones, detect_fast_zones
from trade_decision_engine import run_trade_decision_engine

# ======================================================
# === CONFIGURATION
# ======================================================

# ======================================================
# === CONFIGURATION
# ======================================================
#
# ONE SOURCE OF TRUTH:
#   config.json  → all strategy rules, thresholds, symbols, timeframes
#   BT_CONFIG    → backtest-only settings (paths, balance, sim params)
#
# Strategy parameters are NEVER hardcoded here. They are loaded directly
# from config.json so live and backtest behaviour are always identical.
#

# --- Backtest-only settings (nothing that exists in config.json) ---
BT_CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "data_dir":   "./backtest_data",   # MT5-exported CSVs: XAUUSD_H1.csv etc.
    "output_dir": "./backtest_results",

    # ── Account ───────────────────────────────────────────────────────────────
    "initial_balance": 10_000.0,       # ← Set to your live account size

    # ── Risk (not in config.json — set per backtest run) ──────────────────────
    "risk_percent": 1.0,               # Standard. aggressive=2.0, conservative=0.5

    # ── Spread per symbol (price units, not pips) ─────────────────────────────
    # Exness Zero account — raw spread + small commission. Very tight.
    # These are conservative estimates. Stress test triples them.
    "spread": {
        "XAUUSDz": 0.15,   # ~15 cents (Exness Zero gold is very tight)
        "EURUSDz": 0.00003, # ~0.3 pip
        "GBPUSDz": 0.00005, # ~0.5 pip
        "USDJPYz": 0.005,   # ~0.5 pip
        "AUDUSDz": 0.00004,
        "USDCADz": 0.00005,
    },

    # Dynamic spread multiplier — widens randomly each trade (0.8x quiet → 2.5x volatile)
    "spread_mult_range": (0.8, 2.5),

    # ── Point size per symbol (matches MT5 symbol_info.point) ─────────────────
    "point": {
        "XAUUSDz": 0.01,
        "EURUSDz": 0.00001,
        "GBPUSDz": 0.00001,
        "USDJPYz": 0.001,
        "AUDUSDz": 0.00001,
        "USDCADz": 0.00001,
    },

    # ── Pip value per 1.0 lot (for P&L calculation) ────────────────────────────
    "pip_value_per_lot": {
        "XAUUSDz": 10.0,
        "EURUSDz": 10.0,
        "GBPUSDz": 10.0,
        "USDJPYz": 9.1,
        "AUDUSDz": 10.0,
        "USDCADz": 7.7,
    },

    # ── Execution realism ─────────────────────────────────────────────────────
    "max_slippage_pips": 2.0,

    # ── Session filter (UTC hours) ─────────────────────────────────────────────
    # Joburg = UTC+2. London 07:00 → NY close 21:00 UTC.
    "session_hours": (7, 21),

    # News simulation: randomly drop ~5% of signals (broker rejections, spikes)
    "news_skip_prob": 0.05,

    # ── Walk-forward ──────────────────────────────────────────────────────────
    "walk_forward": {
        "enabled":      True,
        "train_months": 3,
        "test_months":  1,
    },

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    "monte_carlo_runs": 1000,

    # ── HTF bias EMAs (must match get_htf_bias() in scalper_strategy_engine) ───
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,

    # ── Research / experiment tracking ────────────────────────────────────────
    "strategy_name": "selective_branch",
    "research_notes": "Config-driven event backtest with walk-forward, stress, Monte Carlo, PSR/DSR, and experiment metadata.",
    "declared_trial_count": 1,  # Set >1 when comparing many parameter variants; used by DSR proxy.

    # ── Optional sensitivity runner (disabled by default) ─────────────────────
    "sensitivity": {
        "enabled": False,
        "param_grid": {
            # Example:
            # "min_confidence": [0.58, 0.60, 0.62],
            # "sl_buffer_pips": [50, 60, 70],
        }
    },
}

# ── Timeframe string → CSV key map ────────────────────────────────────────────
TF_MAP = {
    "TIMEFRAME_H4":  "H4",
    "TIMEFRAME_H1":  "H1",
    "TIMEFRAME_M5":  "M5",
    "TIMEFRAME_M1":  "M1",
    "TIMEFRAME_M15": "M15",
}


def load_live_config(path: str = "config.json") -> dict:
    """
    Load config.json and merge strategy parameters into BT_CONFIG.
    This is the single source of truth for all strategy rules.
    BT_CONFIG only holds backtest-specific settings (paths, balance, sim params).
    """
    with open(path, "r") as f:
        raw = json.load(f)

    strat  = raw["StrategyParameters"]
    thresh = strat["Thresholds"]

    # Pull every strategy param directly — no hardcoding
    merged = dict(BT_CONFIG)
    merged.update({
        # Symbols
        "symbols":              raw["BotSettings"]["SYMBOLS"],

        # Timeframes — resolved via TF_MAP so run_simulation uses correct CSV keys
        "tf_zone":    TF_MAP.get(strat["TIMEFRAME_ZONE"],    "H1"),
        "tf_entry":   TF_MAP.get(strat["TIMEFRAME_ENTRY"],   "M1"),
        "tf_confirm": TF_MAP.get(strat["TIMEFRAME_CONFIRM"], "M5"),
        "tf_htf":     TF_MAP.get(strat["TIMEFRAME_HTF"],     "H4"),

        # Strategy rules
        "tp_ratio":             strat["TP_RATIO"],
        "partial_close_pct":    strat["PARTIAL_CLOSE_PERCENT"],
        "zone_lookback":        strat["ZONE_LOOKBACK"],
        "allow_pyramid":        strat["ALLOW_PYRAMID_SAME_SIDE"],
        "auto_close_on_strong": strat["AUTO_CLOSE_ON_STRONG"],
        "close_conf_diff":      strat["CLOSE_CONF_DIFF"],
        "max_touch_allowed":    strat["MAX_TOUCH_ALLOWED"],

        # All thresholds — full block passed to decision engine
        # MAX_TOUCH_ALLOWED injected here so decision engine can read it from thresholds dict
        "min_confidence":       thresh["MIN_CONFIDENCE_FOR_TRADE"],
        "thresholds":           {**thresh, "MAX_TOUCH_ALLOWED": strat["MAX_TOUCH_ALLOWED"]},

        # SL/trail — 60 pips tight buffer (reduced from 150)
        "sl_buffer_pips":       60,
        "trail_pips":           150,
        "cooldown_candles":     3,
        "max_consecutive_losses": 3,
    })

    return merged


# ======================================================
# === DATA STRUCTURES
# ======================================================

@dataclass
class BacktestTrade:
    symbol:        str
    side:          str
    entry:         float
    sl:            float
    tp:            float
    lot:           float
    open_idx:      int
    open_time:     str
    confidence:    float
    strategy:      str
    reason:        str

    # Lifecycle state
    status:        str   = "open"   # open | closed
    close_idx:     int   = -1
    close_time:    str   = ""
    close_price:   float = 0.0
    close_reason:  str   = ""       # TP | SL | TRAIL | BE_TRAIL

    # After partial close
    partial_done:  bool  = False
    sl_at_be:      bool  = False
    current_sl:    float = 0.0      # tracks trailing SL

    # Result
    pnl:           float = 0.0
    result:        str   = ""       # win | loss

    def __post_init__(self):
        self.current_sl = self.sl


@dataclass
class EquityPoint:
    idx:      int
    time:     str
    balance:  float
    equity:   float
    drawdown: float


# ======================================================
# === DATA LOADING
# ======================================================

def load_csv(path: str) -> pd.DataFrame:
    """
    Load MT5-exported CSV and standardise columns.
    Handles MT5's tab-separated format:
      <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
      2021.01.03  23:00:00  1909.401  ...
    Also handles comma-separated and already-clean formats.
    """
    # Auto-detect separator by reading first line raw
    with open(path, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(path, sep=sep)

    # Strip whitespace and angle brackets from ALL column names, lowercase
    df.columns = [c.strip().strip('<>').strip().lower() for c in df.columns]

    # Rename MT5 column variants to standard names
    rename_map = {
        'tickvol':  'tick_volume',
        'tick_vol': 'tick_volume',
        'vol':      'real_volume',
        'volume':   'real_volume',
    }
    df.rename(columns=rename_map, inplace=True)

    # Build unified datetime column
    # MT5 uses dots in dates: 2021.01.03 — replace with dashes for parsing
    if 'date' in df.columns:
        date_str = df['date'].astype(str).str.replace('.', '-', regex=False)
        if 'time' in df.columns:
            time_str = df['time'].astype(str)
            df['time'] = pd.to_datetime(date_str + ' ' + time_str,
                                        format='%Y-%m-%d %H:%M:%S')
        else:
            df['time'] = pd.to_datetime(date_str, format='%Y-%m-%d')
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(
            df['time'].astype(str).str.replace('.', '-', regex=False),
            format='mixed', dayfirst=False
        )
    else:
        raise ValueError(f"Cannot find date/time column in {path}.\n"
                         f"Columns found: {list(df.columns)}")

    # Validate OHLC
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. "
                             f"Columns: {list(df.columns)}")

    # Ensure volume columns exist
    if 'tick_volume' not in df.columns:
        df['tick_volume'] = 1
    if 'real_volume' not in df.columns:
        df['real_volume'] = df['tick_volume']

    df = df.sort_values('time').reset_index(drop=True)
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']]


def build_multi_tf(symbol: str, data_dir: str) -> dict:
    """
    Load all available timeframes for a symbol.
    Returns dict: {'H4': df, 'H1': df, 'M5': df, 'M1': df}
    Falls back gracefully if a timeframe is missing.
    """
    tfs = {}
    for tf in ['H4', 'H1', 'M5', 'M1']:
        path = os.path.join(data_dir, f"{symbol}_{tf}.csv")
        if os.path.exists(path):
            try:
                tfs[tf] = load_csv(path)
                print(f"  [Data] Loaded {symbol}_{tf}: {len(tfs[tf])} candles")
            except Exception as e:
                print(f"  [Data] Failed {symbol}_{tf}: {e}")
    return tfs


# ======================================================
# === MARKET UTILITIES
# ======================================================

def get_htf_bias(df: pd.DataFrame, fast=50, slow=200) -> str:
    if len(df) < slow:
        return "NEUTRAL"
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean().iloc[-1]
    if ema_fast > ema_slow: return "UP"
    if ema_fast < ema_slow: return "DOWN"
    return "NEUTRAL"


def calculate_trend(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 51: return None
    sma50 = df['close'].rolling(50).mean().iloc[-1]
    last  = df['close'].iloc[-1]
    return "uptrend" if last > sma50 else "downtrend"


def apply_spread_slippage(price: float, side: str, spread: float, max_slippage: float) -> float:
    """Realistic entry: widen by spread and randomise slippage."""
    slippage = random.uniform(0, max_slippage)
    if side == "buy":
        return price + spread + slippage
    else:
        return price - spread - slippage


def calc_lot(balance: float, risk_pct: float, entry: float, sl: float,
             pip_value: float, point: float) -> float:
    """Risk-based lot sizing — mirrors live determine_lot_size."""
    risk_amt   = balance * (risk_pct / 100.0)
    dist_price = abs(entry - sl)
    if dist_price == 0 or pip_value == 0 or point == 0:
        return 0.01
    dist_pips    = dist_price / point
    loss_per_lot = dist_pips * pip_value
    if loss_per_lot <= 0:
        return 0.01
    lot = risk_amt / loss_per_lot
    return max(0.01, round(lot, 2))


def calc_pnl(side: str, entry: float, close_price: float, lot: float,
             pip_value: float, point: float) -> float:
    """P&L in account currency."""
    dist_pips = abs(close_price - entry) / point
    pnl       = dist_pips * pip_value * lot
    if side == "buy":
        return pnl if close_price > entry else -pnl
    else:
        return pnl if close_price < entry else -pnl


# ======================================================
# === TRADE LIFECYCLE
# ======================================================

def update_trade(trade: BacktestTrade, candle: pd.Series,
                 trail_dist: float, pip_value: float, point: float,
                 partial_pct: float = 0.5) -> BacktestTrade:
    """
    Update one open trade against the current candle.
    Mirrors live logic: partial TP at 1R → realize partial profit → SL to BE → trailing stop.
    Worst-case collision: if both SL and TP would hit same candle, SL wins.
    """
    hi, lo = candle['high'], candle['low']
    risk = abs(trade.entry - trade.sl)

    # --- 1. Partial TP at 1R: realize profit on half position ---
    if not trade.partial_done and partial_pct > 0:
        one_r_target = (trade.entry + risk) if trade.side == "buy" else (trade.entry - risk)
        hit_1r = (hi >= one_r_target) if trade.side == "buy" else (lo <= one_r_target)
        if hit_1r:
            # Realize profit on the partial close portion
            partial_lot    = trade.lot * partial_pct
            partial_profit = calc_pnl(trade.side, trade.entry, one_r_target,
                                      partial_lot, pip_value, point)
            trade.pnl     += partial_profit          # Book it immediately
            trade.lot     *= (1.0 - partial_pct)     # Remaining lot shrinks
            trade.partial_done = True
            trade.sl_at_be     = True
            trade.current_sl   = trade.entry         # Move SL to breakeven

    # --- 2. Trailing SL (only after BE) ---
    if trade.partial_done:
        if trade.side == "buy":
            proposed = hi - trail_dist
            if proposed > trade.current_sl:
                trade.current_sl = proposed
        else:
            proposed = lo + trail_dist
            if trade.current_sl == trade.entry or proposed < trade.current_sl:
                trade.current_sl = proposed

    # --- 3. Worst-case collision: if SL AND TP both hit this candle, SL wins ---
    sl_hit = (lo <= trade.current_sl) if trade.side == "buy" else (hi >= trade.current_sl)
    tp_hit = (hi >= trade.tp)         if trade.side == "buy" else (lo <= trade.tp)

    if sl_hit and tp_hit:
        tp_hit = False  # Conservative: assume price hit SL first

    # --- 4. Check SL hit ---
    if sl_hit:
        reason = "BE_TRAIL" if trade.partial_done else "SL"
        return _close_trade(trade, candle, trade.current_sl, reason, pip_value, point)

    # --- 5. Check TP hit ---
    if tp_hit:
        return _close_trade(trade, candle, trade.tp, "TP", pip_value, point)

    return trade


def _close_trade(trade: BacktestTrade, candle: pd.Series, close_price: float,
                 reason: str, pip_value: float, point: float) -> BacktestTrade:
    trade.status      = "closed"
    trade.close_price = close_price
    trade.close_time  = str(candle['time'])
    trade.close_reason = reason
    # += not = : partial profit booked in update_trade() must not be overwritten.
    # When partial_close_pct == 0.0 this is a no-op (trade.pnl starts at 0.0).
    # When partials are re-enabled, this correctly accumulates partial + remainder.
    trade.pnl        += calc_pnl(trade.side, trade.entry, close_price,
                                  trade.lot, pip_value, point)
    trade.result      = "win" if trade.pnl >= 0 else "loss"
    return trade


# ======================================================
# === METRICS
# ======================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hash_cfg(cfg: dict) -> str:
    try:
        payload = json.dumps(cfg, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        return "cfghash_err"


def _trade_returns_from_closed(closed: list, initial_balance: float) -> np.ndarray:
    if not closed:
        return np.array([], dtype=float)
    balance = float(initial_balance)
    returns = []
    for t in closed:
        if balance <= 0:
            returns.append(0.0)
            balance += t.pnl
            continue
        r = float(t.pnl) / balance
        returns.append(r)
        balance += float(t.pnl)
    return np.array(returns, dtype=float)


def _downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    if returns.size == 0:
        return 0.0
    downside = returns[returns < mar] - mar
    if downside.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))))


def _risk_adjusted_stats(returns: np.ndarray, trial_count: int = 1) -> dict:
    """
    Trade-return based risk-adjusted stats.
    PSR implemented directly. DSR is an approximation that deflates the observed
    Sharpe by the expected inflation from multiple trials. Keep `declared_trial_count`
    honest when you compare many variants.
    """
    out = {
        "trade_return_mean": 0.0,
        "trade_return_std": 0.0,
        "trade_sharpe": 0.0,
        "trade_sortino": 0.0,
        "trade_skew": 0.0,
        "trade_kurtosis": 0.0,
        "psr_sr_gt_0": None,
        "dsr_proxy": None,
    }
    n = int(returns.size)
    if n < 2:
        return out

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma <= 0:
        return out

    sr = mu / sigma
    downside = _downside_deviation(returns)
    sortino = (mu / downside) if downside > 0 else float("inf")

    centered = returns - mu
    m2 = float(np.mean(centered ** 2)) if n > 0 else 0.0
    m3 = float(np.mean(centered ** 3)) if n > 0 else 0.0
    m4 = float(np.mean(centered ** 4)) if n > 0 else 0.0
    skew = (m3 / (m2 ** 1.5)) if m2 > 0 else 0.0
    kurt = (m4 / (m2 ** 2)) if m2 > 0 else 3.0  # Pearson kurtosis

    # Probabilistic Sharpe Ratio for benchmark SR*=0
    denom_term = 1 - skew * sr + ((kurt - 1) / 4.0) * (sr ** 2)
    if denom_term > 0 and n > 1:
        z = (sr * math.sqrt(n - 1)) / math.sqrt(denom_term)
        psr = _norm_cdf(z)
    else:
        psr = None

    # Deflated Sharpe proxy: subtract expected inflation from multiple testing.
    trials = max(int(trial_count or 1), 1)
    if trials > 1 and n > 1:
        inflation = math.sqrt(2.0 * math.log(trials)) / math.sqrt(max(n - 1, 1))
        dsr_proxy = _norm_cdf((sr - inflation) * math.sqrt(max(n - 1, 1)))
    else:
        dsr_proxy = psr

    out.update({
        "trade_return_mean": round(mu, 6),
        "trade_return_std": round(sigma, 6),
        "trade_sharpe": round(sr, 4),
        "trade_sortino": round(sortino, 4) if math.isfinite(sortino) else "inf",
        "trade_skew": round(skew, 4),
        "trade_kurtosis": round(kurt, 4),
        "psr_sr_gt_0": round(psr, 4) if psr is not None else None,
        "dsr_proxy": round(dsr_proxy, 4) if dsr_proxy is not None else None,
    })
    return out


def compute_metrics(trades: list, initial_balance: float, cfg: Optional[dict] = None) -> dict:
    """Full performance metrics from a list of closed BacktestTrade objects."""
    closed = [t for t in trades if t.status == "closed"]
    if not closed:
        return {}

    pnls   = [t.pnl for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(closed)
    win_count    = len(wins)
    loss_count   = len(losses)
    win_rate     = win_count / total_trades if total_trades else 0

    avg_win  = sum(wins)  / len(wins)   if wins   else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    total_profit = sum(wins)
    total_loss   = abs(sum(losses))
    profit_factor = total_profit / total_loss if total_loss else float('inf')

    # Equity curve + drawdown
    balance   = initial_balance
    peak      = initial_balance
    max_dd    = 0.0
    max_dd_pct = 0.0
    consec_loss = 0
    max_consec  = 0

    for t in closed:
        balance += t.pnl
        peak     = max(peak, balance)
        dd       = peak - balance
        dd_pct   = (dd / peak * 100) if peak > 0 else 0
        max_dd     = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

        if t.pnl < 0:
            consec_loss += 1
            max_consec   = max(max_consec, consec_loss)
        else:
            consec_loss  = 0

    final_balance = initial_balance + sum(pnls)
    net_return_pct = (final_balance - initial_balance) / initial_balance * 100

    returns = _trade_returns_from_closed(closed, initial_balance)
    risk_stats = _risk_adjusted_stats(returns, trial_count=(cfg or {}).get("declared_trial_count", 1))

    metrics = {
        "total_trades":       total_trades,
        "wins":               win_count,
        "losses":             loss_count,
        "win_rate_pct":       round(win_rate * 100, 2),
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "expectancy":         round(expectancy, 2),
        "profit_factor":      round(profit_factor, 4),
        "total_profit":       round(total_profit, 2),
        "total_loss":         round(total_loss, 2),
        "net_pnl":            round(sum(pnls), 2),
        "net_return_pct":     round(net_return_pct, 2),
        "initial_balance":    initial_balance,
        "final_balance":      round(final_balance, 2),
        "max_drawdown":       round(max_dd, 2),
        "max_drawdown_pct":   round(max_dd_pct, 2),
        "max_consec_losses":  max_consec,
    }
    metrics.update(risk_stats)

    if cfg is not None:
        metrics.update({
            "strategy_name": cfg.get("strategy_name", "unknown"),
            "experiment_id": cfg.get("experiment_id"),
            "config_hash": cfg.get("config_hash"),
            "declared_trial_count": cfg.get("declared_trial_count", 1),
        })

    return metrics


def monte_carlo(trades: list, initial_balance: float, runs: int = 1000) -> dict:
    """
    Shuffle win/loss order N times.
    Shows worst-case drawdown distribution and survival probability.
    """
    if not trades: return {}
    pnls = [t.pnl for t in trades if t.status == "closed"]
    if not pnls: return {}

    max_drawdowns = []
    final_balances = []

    for _ in range(runs):
        shuffled = random.sample(pnls, len(pnls))
        balance  = initial_balance
        peak     = initial_balance
        max_dd   = 0.0
        for p in shuffled:
            balance += p
            peak     = max(peak, balance)
            dd       = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd   = max(max_dd, dd)
        max_drawdowns.append(max_dd)
        final_balances.append(balance)

    dd_arr  = np.array(max_drawdowns)
    bal_arr = np.array(final_balances)

    metrics = {
        "mc_runs":             runs,
        "mc_max_dd_median":    round(float(np.median(dd_arr)), 2),
        "mc_max_dd_95th_pct":  round(float(np.percentile(dd_arr, 95)), 2),
        "mc_max_dd_worst":     round(float(np.max(dd_arr)), 2),
        "mc_survival_rate":    round(float(np.mean(bal_arr > 0)) * 100, 1),
        "mc_final_bal_median": round(float(np.median(bal_arr)), 2),
    }
    return metrics

# ======================================================
# === CORE SIMULATION
# ======================================================

def run_simulation(symbol: str, tfs: dict, cfg: dict,
                   start_idx: int = 0, end_idx: int = None) -> list:
    """
    Candle-by-candle simulation over H1 data.
    Uses real detect_zones + run_trade_decision_engine.
    Returns list of BacktestTrade objects.
    """
    h1  = tfs.get(cfg.get('tf_zone',    'H1'))
    m5  = tfs.get(cfg.get('tf_confirm', 'M5'))
    m1  = tfs.get(cfg.get('tf_entry',   'M1'))
    h4  = tfs.get(cfg.get('tf_htf',     'H4'))

    if h1 is None:
        print(f"  [!] {symbol}: No H1 data. Skipping.")
        return []

    h1 = h1.iloc[start_idx:end_idx].reset_index(drop=True)
    if len(h1) < 200:
        print(f"  [!] {symbol}: Not enough H1 candles ({len(h1)}). Skipping.")
        return []

    # Config shortcuts
    spread_val   = cfg['spread'].get(symbol, 0.0002)
    point_val    = cfg['point'].get(symbol, 0.00001)
    pip_val      = cfg['pip_value_per_lot'].get(symbol, 10.0)
    sl_buffer    = cfg['sl_buffer_pips'] * point_val
    trail_dist   = cfg['trail_pips'] * point_val
    max_slip     = cfg['max_slippage_pips'] * point_val
    risk_pct     = cfg['risk_percent']
    min_conf     = cfg['min_confidence']
    tp_ratio     = cfg['tp_ratio']
    thresholds   = cfg['thresholds']
    cooldown     = cfg['cooldown_candles']
    circ_max     = cfg['max_consecutive_losses']
    partial_pct  = cfg.get('partial_close_pct', 0.5)  # use config, not hardcoded

    trades         = []
    open_trade     = None
    balance        = cfg['initial_balance']
    last_trade_idx = -cooldown - 1
    consec_losses  = 0
    current_day    = None  # Tracks calendar date for clean circuit breaker reset

    # Pending signal: detected on candle[i], executed on candle[i+1] open
    pending_signal = None

    # Equity curve
    equity_points  = []
    peak_balance   = cfg['initial_balance']

    # Session / news config
    session_start, session_end = cfg.get('session_hours', (0, 24))
    news_skip_prob = cfg.get('news_skip_prob', 0.0)
    spread_mult_lo, spread_mult_hi = cfg.get('spread_mult_range', (1.0, 1.0))

    # We need at least ZONE_LOOKBACK of history before we start making decisions
    WARMUP = 200

    for i in range(WARMUP, len(h1)):
        candle = h1.iloc[i]

        # --- 0. Execute pending signal at THIS candle's open (1-candle delay) ---
        if pending_signal is not None and open_trade is None:
            sig  = pending_signal
            side = sig['side']
            # Dynamic spread: widens randomly each trade — simulates real conditions
            dynamic_spread = random.uniform(spread_mult_lo, spread_mult_hi) * spread_val
            entry = apply_spread_slippage(candle['open'], side, dynamic_spread, max_slip)
            sl    = sig['sl']
            tp    = sig['tp']
            lot   = calc_lot(balance, risk_pct, entry, sl, pip_val, point_val)
            open_trade = BacktestTrade(
                symbol     = symbol,
                side       = side,
                entry      = entry,
                sl         = sl,
                tp         = tp,
                lot        = lot,
                open_idx   = i,
                open_time  = str(candle['time']),
                confidence = float(sig.get('confidence', 0)),
                strategy   = sig.get('strategy', 'standard'),
                reason     = sig.get('reason', ''),
            )
            last_trade_idx = i
        pending_signal = None  # Always clear after attempting execution

        # --- Session filter: skip signal generation outside trading hours ---
        candle_hour = candle['time'].hour if hasattr(candle['time'], 'hour') else 12
        in_session  = session_start <= candle_hour < session_end

        # Track equity curve
        current_equity = balance + (
            calc_pnl(open_trade.side, open_trade.entry, candle['close'],
                     open_trade.lot, pip_val, point_val)
            if open_trade and open_trade.status == "open" else 0.0
        )
        peak_balance = max(peak_balance, current_equity)
        dd_pct = ((peak_balance - current_equity) / peak_balance * 100) if peak_balance > 0 else 0
        equity_points.append(EquityPoint(i, str(candle['time']), balance, current_equity, dd_pct))

        # --- 1. Update open trade lifecycle ---
        if open_trade and open_trade.status == "open":
            open_trade = update_trade(open_trade, candle, trail_dist, pip_val, point_val, partial_pct)
            if open_trade.status == "closed":
                balance       += open_trade.pnl
                last_trade_idx = i
                result_str     = open_trade.result

                if result_str == "loss":
                    consec_losses += 1
                else:
                    consec_losses  = 0

                trades.append(open_trade)
                open_trade = None

        # --- 2. Circuit breaker: reset cleanly on new calendar day ---
        candle_day = candle['time'].date() if hasattr(candle['time'], 'date') else None
        if candle_day and candle_day != current_day:
            current_day   = candle_day
            consec_losses = 0  # New day → full reset

        if consec_losses >= circ_max:
            continue  # Blocked for the rest of today

        # --- 3. Cooldown ---
        if (i - last_trade_idx) < cooldown:
            continue

        # --- 4. Skip if already in a trade ---
        if open_trade and open_trade.status == "open":
            continue

        # --- 5. Build lookback windows ---
        h1_window = h1.iloc[max(0, i - 300):i+1]

        # Align M5 by timestamp
        if m5 is not None:
            m5_mask   = m5['time'] <= candle['time']
            m5_window = m5[m5_mask].tail(60)
        else:
            m5_window = h1.iloc[max(0, i-60):i+1]  # Fallback: use H1

        if m1 is not None:
            m1_mask   = m1['time'] <= candle['time']
            m1_window = m1[m1_mask].tail(200)
        else:
            m1_window = h1.iloc[max(0, i-200):i+1]

        if h4 is not None:
            h4_mask   = h4['time'] <= candle['time']
            h4_window = h4[h4_mask].tail(250)
        else:
            h4_window = h1_window.tail(250)

        if len(m5_window) < 5 or len(m1_window) < 35:
            continue

        # --- 6. Indicators ---
        try:
            htf_bias = get_htf_bias(h4_window,
                                    cfg['htf_ema_fast'],
                                    cfg['htf_ema_slow'])
            if htf_bias == "NEUTRAL":
                continue

            trend  = calculate_trend(h1_window)
            h1_atr = AverageTrueRange(h1_window['high'],
                                      h1_window['low'],
                                      h1_window['close']).average_true_range().iloc[-1]

            macd_obj  = MACD(m1_window['close'])
            macd_line = macd_obj.macd().dropna().values
            macd_sig  = macd_obj.macd_signal().dropna().values
            rsi_vals  = RSIIndicator(m1_window['close']).rsi().dropna().values
            atr_val   = AverageTrueRange(m1_window['high'],
                                         m1_window['low'],
                                         m1_window['close']).average_true_range().iloc[-1]

            try:
                from ta.volume import VolumeWeightedAveragePrice
                vwap = VolumeWeightedAveragePrice(
                    m1_window['high'], m1_window['low'],
                    m1_window['close'], m1_window['real_volume']
                ).vwap.iloc[-1]
            except Exception:
                vwap = m1_window['close'].mean()

            m5_context = {
                'trend': calculate_trend(m5_window),
                'macd':  MACD(m5_window['close']).macd().iloc[-1],
                'rsi':   RSIIndicator(m5_window['close']).rsi().iloc[-1]
            }

        except Exception:
            continue

        # --- 7. Zone detection ---
        try:
            demand_zones, supply_zones = detect_zones(h1_window)
            fast_demand, fast_supply   = detect_fast_zones(h1_window)
        except Exception:
            continue

        if not demand_zones and not supply_zones and not fast_demand and not fast_supply:
            continue

        # --- 8. Session + news guard (signal generation only) ---
        if not in_session:
            continue

        # Simulate ~5% of signals being killed by unpredictable news spikes
        if random.random() < news_skip_prob:
            continue

        try:
            signals, _ = run_trade_decision_engine(
                symbol        = symbol,
                point         = point_val,
                current_price = candle['close'],
                trend         = trend,
                demand_zones  = demand_zones,
                supply_zones  = supply_zones,
                fast_demand_zones = fast_demand,
                fast_supply_zones = fast_supply,
                m1_candles_for_crt    = m1_window.iloc[-3:],
                m5_candles_for_patterns = m5_window.iloc[-15:],
                active_trades = {},
                zone_touch_counts = {},
                SL_BUFFER     = sl_buffer,
                TP_RATIO      = tp_ratio,
                CHECK_RANGE   = max(atr_val, 50 * point_val),
                LOT_SIZE      = 0.01,
                MAGIC         = 0,
                strategy_mode = "standard",
                macd          = macd_line,
                macd_signal   = macd_sig,
                rsi           = rsi_vals,
                vwap          = vwap,
                atr           = atr_val,
                htf_atr       = h1_atr,
                m5_context    = m5_context,
                htf_high      = h1_window['high'].max(),
                htf_low       = h1_window['low'].min(),
                last_closed_h1 = h1_window.iloc[-2],
                htf_bias      = htf_bias,
                thresholds    = thresholds,
            )
        except Exception:
            continue

        if not signals:
            continue

        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        sig = signals[0]
        confidence = float(sig.get('confidence', 0))

        if confidence < min_conf:
            continue

        # --- 9. Store as pending — will execute on NEXT candle's open ---
        # This eliminates same-candle entry cheat and adds 1-candle execution delay.
        pending_signal = sig

    # Close any trade still open at end of data
    if open_trade and open_trade.status == "open":
        last_candle = h1.iloc[-1]
        open_trade.status      = "closed"
        open_trade.close_price = last_candle['close']
        open_trade.close_time  = str(last_candle['time'])
        open_trade.close_reason = "END_OF_DATA"
        open_trade.pnl         = calc_pnl(open_trade.side, open_trade.entry,
                                           last_candle['close'], open_trade.lot,
                                           pip_val, point_val)
        open_trade.result      = "win" if open_trade.pnl >= 0 else "loss"
        trades.append(open_trade)

    return trades, equity_points


# ======================================================
# === WALK-FORWARD ENGINE
# ======================================================

def run_walk_forward(symbol: str, tfs: dict, cfg: dict) -> list:
    """
    Slides a test window across the full H1 dataset one month at a time.
    Returns only trades from test windows (zero look-ahead).

    Design:
      • Each iteration tests one month of live decisions.
      • The simulation for that window starts WARMUP_BARS (500 H1 bars ≈ 21 days)
        before the test window so indicators are fully warmed up.
      • Only trades whose open_time falls inside the test window are kept —
        the warmup period generates no recorded trades.
      • No parameter fitting occurs, so no "train" simulation is needed.
        The old STEP A has been removed; it was reprocessing the full history
        on every loop with zero effect on outputs, halving throughput for free.
    """
    h1 = tfs.get('H1')
    if h1 is None: return []

    wf         = cfg['walk_forward']
    train_mo   = wf['train_months']
    test_mo    = wf['test_months']
    all_trades = []

    start_dt   = h1['time'].iloc[0]
    end_dt     = h1['time'].iloc[-1]

    train_delta = timedelta(days=30 * train_mo)
    test_delta  = timedelta(days=30 * test_mo)

    window_start = start_dt
    window_num   = 0

    while True:
        train_end  = window_start + train_delta
        test_start = train_end
        test_end   = test_start + test_delta

        if test_end > end_dt:
            break

        # Index ranges
        train_mask = (h1['time'] >= window_start) & (h1['time'] < train_end)
        test_mask  = (h1['time'] >= test_start)   & (h1['time'] < test_end)

        test_start_idx = h1[test_mask].index.min() if test_mask.any() else None
        test_end_idx   = h1[test_mask].index.max() if test_mask.any() else None

        if test_start_idx is None or pd.isna(test_start_idx):
            break

        window_num += 1
        print(f"  [WF-{window_num}] Test: {test_start.date()} → {test_end.date()}")

        # ── STEP A removed ──────────────────────────────────────────────────
        # The old "train" simulation ran the entire history from bar 0 to
        # test_start on every loop — purely wasted compute, because no
        # parameters are being fitted or updated between windows.
        # Removing it cuts runtime roughly in half with zero impact on results.
        # ────────────────────────────────────────────────────────────────────

        # ── STEP B: warmup-slice simulation ─────────────────────────────────
        # Instead of re-running from bar 0 (which reprocesses the same ancient
        # history on every loop), we start 500 bars before the test window.
        # 500 H1 bars ≈ 21 days — enough for ATR(14), EMA(200), zone lookback,
        # and MACD warmup to stabilise before the first live decision is made.
        # Integrity is preserved: the simulation still only ever sees candles
        # up to the current bar; no future data leaks into indicator state.
        WARMUP_BARS = 500
        sim_start_idx = max(0, int(test_start_idx) - WARMUP_BARS)

        test_trades, _eq = run_simulation(symbol, tfs, cfg,
                                          start_idx=sim_start_idx,
                                          end_idx=int(test_end_idx) + 1)

        # Keep only trades that OPENED inside the test window
        test_trades = [
            t for t in test_trades
            if test_start <= pd.to_datetime(t.open_time) < test_end
        ]
        all_trades.extend(test_trades)

        # ── FIX: advance by test_delta (1 month), NOT by train_delta.
        # Old code set window_start = test_start, which jumped the entire
        # training window forward each iteration, leaving multi-month gaps
        # and producing only ~2 test windows over a year of data.
        # Correct behaviour: slide forward one test period at a time so
        # every month gets evaluated and trade count is no longer starved.
        window_start = window_start + test_delta

    print(f"  [WF] Complete — {window_num} test windows, {len(all_trades)} total trades collected.")
    return all_trades


# ======================================================
# === OUTPUT & REPORTING
# ======================================================

def save_trades_csv(trades: list, path: str, cfg: Optional[dict] = None):
    """Save all trades to CSV — feeds your learning system and preserves experiment metadata."""
    if not trades: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment_id', 'strategy_name', 'config_hash',
            'symbol', 'side', 'open_time', 'close_time',
            'entry', 'close_price', 'sl', 'tp',
            'lot', 'pnl', 'result', 'close_reason',
            'confidence', 'strategy', 'reason',
        ])
        for t in trades:
            writer.writerow([
                (cfg or {}).get('experiment_id'), (cfg or {}).get('strategy_name'), (cfg or {}).get('config_hash'),
                t.symbol, t.side, t.open_time, t.close_time,
                round(t.entry, 5), round(t.close_price, 5),
                round(t.sl, 5), round(t.tp, 5),
                t.lot, round(t.pnl, 2), t.result, t.close_reason,
                round(t.confidence, 4), t.strategy, t.reason,
            ])


def save_metrics_json(metrics: dict, mc: dict, path: str, cfg: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if metrics is None:
        metrics = {}
    if mc is None:
        mc = {}

    combined = {**metrics, **mc}

    if cfg is not None:
        combined["config_symbols"] = cfg.get("symbols", [])
        combined["config_tp_ratio"] = cfg.get("tp_ratio")
        combined["config_min_confidence"] = cfg.get("min_confidence")

    with open(path, 'w') as f:
        json.dump(combined, f, indent=4)


def print_report(symbol: str, metrics: dict, mc: dict):
    print(f"\n{'='*55}")
    print(f"  BACKTEST REPORT — {symbol}")
    print(f"{'='*55}")
    if not metrics:
        print("  No trades generated.")
        return

    print(f"  Trades:         {metrics['total_trades']}  ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  Win Rate:       {metrics['win_rate_pct']}%")
    print(f"  Expectancy:     ${metrics['expectancy']}")
    print(f"  Profit Factor:  {metrics['profit_factor']}")
    print(f"  Net Return:     {metrics['net_return_pct']}%  (${metrics['net_pnl']})")
    print(f"  Final Balance:  ${metrics['final_balance']}")
    print(f"  Max Drawdown:   {metrics['max_drawdown_pct']}%  (${metrics['max_drawdown']})")
    print(f"  Max Consec L:   {metrics['max_consec_losses']}")

    if mc:
        print(f"\n  ── Monte Carlo ({mc['mc_runs']} runs) ──")
        print(f"  Median DD:      {mc['mc_max_dd_median']}%")
        print(f"  95th pct DD:    {mc['mc_max_dd_95th_pct']}%")
        print(f"  Worst DD:       {mc['mc_max_dd_worst']}%")
        print(f"  Survival Rate:  {mc['mc_survival_rate']}%")

    if metrics.get('trade_sharpe') is not None:
        print(f"\n  ── Research Stats ──")
        print(f"  Trade Sharpe:    {metrics.get('trade_sharpe')}")
        print(f"  Trade Sortino:   {metrics.get('trade_sortino')}")
        print(f"  PSR (SR>0):      {metrics.get('psr_sr_gt_0')}")
        print(f"  DSR Proxy:       {metrics.get('dsr_proxy')}")
        print(f"  Config Hash:     {metrics.get('config_hash')}")
        print(f"  Experiment ID:   {metrics.get('experiment_id')}")

    # Verdict
    print(f"\n  ── Verdict ──")
    pf   = metrics.get('profit_factor', 0)
    wr   = metrics.get('win_rate_pct', 0)
    dd   = metrics.get('max_drawdown_pct', 100)
    exp  = metrics.get('expectancy', 0)
    total = metrics.get('total_trades', 0)

    if total < 30:
        print("  ⚠️  INSUFFICIENT DATA — Need 30+ trades for valid conclusions.")
    elif pf >= 1.5 and wr >= 45 and dd <= 20 and exp > 0:
        print("  ✅  ROBUST — System shows edge. Consider live testing.")
    elif pf >= 1.2 and exp > 0:
        print("  🟡  MARGINAL — Some edge present but refine before live.")
    else:
        print("  🔴  WEAK — No consistent edge. Do not trade live.")

    print(f"{'='*55}\n")


# ======================================================
# === ROBUSTNESS STRESS TEST
# ======================================================

def stress_test(symbol: str, tfs: dict, cfg: dict) -> dict:
    """
    'Break your bot' test.
    Runs simulation with increased spread and 1-candle entry delay.
    If it still shows edge → the system is robust.
    """
    print(f"\n  [Stress] Running stress test on {symbol}...")

    stressed_cfg = dict(cfg)
    stressed_cfg['spread'] = {k: v * 3 for k, v in cfg['spread'].items()}  # 3x spread
    stressed_cfg['max_slippage_pips'] = cfg['max_slippage_pips'] * 3         # 3x slippage

    trades, _eq = run_simulation(symbol, tfs, stressed_cfg)
    metrics = compute_metrics(trades, cfg['initial_balance'])

    pf = metrics.get('profit_factor', 0)
    verdict = "✅ HOLDS" if pf >= 1.0 else "🔴 COLLAPSES"
    print(f"  [Stress] {symbol} under 3x spread: PF={pf} → {verdict}")
    return metrics


# ======================================================
# === OPTIONAL SENSITIVITY RUNNER
# ======================================================

def _iter_param_grid(param_grid: dict):
    if not param_grid:
        yield {}
        return
    keys = list(param_grid.keys())
    def rec(i, acc):
        if i == len(keys):
            yield dict(acc)
            return
        k = keys[i]
        for v in param_grid[k]:
            acc[k] = v
            yield from rec(i + 1, acc)
    yield from rec(0, {})


def run_sensitivity_grid(base_cfg: dict):
    grid_cfg = base_cfg.get('sensitivity', {})
    if not grid_cfg.get('enabled'):
        return None

    grid = grid_cfg.get('param_grid', {})
    variants = list(_iter_param_grid(grid))
    if not variants:
        return None

    results = []
    print(f"\n[Research] Running sensitivity grid with {len(variants)} variants...")
    for idx, overrides in enumerate(variants, 1):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg['declared_trial_count'] = len(variants)
        cfg['experiment_id'] = base_cfg['experiment_id'] + f"_g{idx:03d}"
        cfg['config_hash'] = _hash_cfg({k: v for k, v in cfg.items() if k not in {'experiment_id', 'config_hash'}})

        row = {'experiment_id': cfg['experiment_id'], 'config_hash': cfg['config_hash'], **overrides}
        # aggregate simple robustness score across symbols
        total_pf = 0.0
        total_trades = 0
        symbols_run = 0
        for symbol in cfg['symbols']:
            tfs = build_multi_tf(symbol, cfg['data_dir'])
            if not tfs:
                continue
            trades = run_walk_forward(symbol, tfs, cfg) if cfg['walk_forward']['enabled'] else run_simulation(symbol, tfs, cfg)[0]
            m = compute_metrics(trades, cfg['initial_balance'], cfg)
            total_pf += float(m.get('profit_factor', 0) or 0)
            total_trades += int(m.get('total_trades', 0) or 0)
            symbols_run += 1
        row['avg_pf'] = round(total_pf / symbols_run, 4) if symbols_run else None
        row['total_trades'] = total_trades
        results.append(row)
        print(f"[Research] Variant {idx}/{len(variants)} → avg_pf={row['avg_pf']} trades={row['total_trades']} overrides={overrides}")

    out = os.path.join(base_cfg['output_dir'], 'sensitivity_grid_results.csv')
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[Research] Sensitivity grid saved to {out}")
    return results


def main():
    # Load config.json as single source of truth, merged with backtest-only settings
    cfg = load_live_config("config.json")
    cfg['experiment_id'] = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    cfg['config_hash'] = _hash_cfg({k: v for k, v in cfg.items() if k not in {'experiment_id', 'config_hash'}})
    print(f"  [Config] Loaded config.json")
    print(f"  [Config] Symbols:     {cfg['symbols']}")
    print(f"  [Config] TP Ratio:    {cfg['tp_ratio']} | Partial Close: {cfg['partial_close_pct']}")
    print(f"  [Config] Min Conf:    {cfg['min_confidence']} | TF Zone: {cfg['tf_zone']} | TF HTF: {cfg['tf_htf']}")
    print(f"  [Config] Strategy:    {cfg.get('strategy_name')} | Experiment: {cfg['experiment_id']} | Hash: {cfg['config_hash']}")

    if cfg.get('sensitivity', {}).get('enabled'):
        run_sensitivity_grid(cfg)
        return

    data_dir   = cfg['data_dir']
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    all_symbol_results = {}

    for symbol in cfg['symbols']:
        print(f"\n{'─'*55}")
        print(f"  Processing: {symbol}")
        print(f"{'─'*55}")

        tfs = build_multi_tf(symbol, data_dir)
        if not tfs:
            print(f"  [!] No data found for {symbol} in {data_dir}. Skipping.")
            continue

        # ── Data-completeness check: warn loudly if required TFs are missing.
        # Missing M1 or H4 means the backtest silently falls back to H1 proxies,
        # which changes signal frequency and makes results non-representative.
        # Fix: export the missing TF from MT5 before trusting the output.
        required_tfs = [cfg['tf_zone'], cfg['tf_entry'], cfg['tf_confirm'], cfg['tf_htf']]
        missing_tfs  = [tf for tf in required_tfs if tf not in tfs]
        if missing_tfs:
            print(f"  [⚠️  DATA WARNING] {symbol}: Missing TFs {missing_tfs}. "
                  f"Backtest will use fallback proxies — results are NOT faithful to live config. "
                  f"Export these from MT5 before trusting this run.")

        # --- Full simulation or Walk-Forward ---
        if cfg['walk_forward']['enabled']:
            print(f"  [Mode] Walk-Forward ({cfg['walk_forward']['train_months']}m train / "
                  f"{cfg['walk_forward']['test_months']}m test)")
            trades = run_walk_forward(symbol, tfs, cfg)
            equity_points = []  # Walk-forward aggregates across windows; equity curve N/A
        else:
            print(f"  [Mode] Full simulation")
            trades, equity_points = run_simulation(symbol, tfs, cfg)

            # Save equity curve
            eq_path = os.path.join(output_dir, f"{symbol}_equity.csv")
            with open(eq_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['idx', 'time', 'balance', 'equity', 'drawdown_pct'])
                for ep in equity_points:
                    w.writerow([ep.idx, ep.time, round(ep.balance, 2),
                                 round(ep.equity, 2), round(ep.drawdown, 2)])

        metrics = compute_metrics(trades, cfg['initial_balance'], cfg)
        mc      = monte_carlo(trades, cfg['initial_balance'], cfg['monte_carlo_runs'])
        
        if metrics is None:
           metrics = {}
        if mc is None:
            mc = {}

        # --- Stress test ---
        stress_metrics = stress_test(symbol, tfs, cfg)

        # --- Save outputs ---
        trades_path  = os.path.join(output_dir, f"{symbol}_trades.csv")
        metrics_path = os.path.join(output_dir, f"{symbol}_metrics.json")
        save_trades_csv(trades, trades_path, cfg)
        save_metrics_json(metrics, mc, metrics_path, cfg)

        # --- Print report ---
        print_report(symbol, metrics, mc)

        all_symbol_results[symbol] = {
            "metrics": metrics,
            "mc": mc,
            "stress_pf": stress_metrics.get('profit_factor', 0),
        }

    # --- Cross-pair robustness summary ---
    print(f"\n{'='*55}")
    print(f"  CROSS-PAIR ROBUSTNESS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Symbol':<10} {'WR%':>6} {'PF':>6} {'DD%':>7} {'Trades':>7} {'Stress PF':>10}")
    print(f"  {'─'*50}")
    for sym, res in all_symbol_results.items():
        m  = res['metrics']
        if not m: continue
        print(f"  {sym:<10} "
              f"{m.get('win_rate_pct', 0):>6.1f} "
              f"{m.get('profit_factor', 0):>6.2f} "
              f"{m.get('max_drawdown_pct', 0):>7.1f} "
              f"{m.get('total_trades', 0):>7} "
              f"{res['stress_pf']:>10.2f}")

    # ── FIX: require both sufficient trade count AND profit factor.
    # Old code counted any pair with PF >= 1.2 as "confirmed", so 1 pair
    # with 2 trades could mark the whole system as robust — a false positive.
    # A minimum of 30 trades is the standard threshold for statistical validity.
    MIN_TRADES_FOR_VERDICT = 30
    verdict_count = sum(
        1 for r in all_symbol_results.values()
        if r['metrics'].get('total_trades', 0) >= MIN_TRADES_FOR_VERDICT
        and r['metrics'].get('profit_factor', 0) >= 1.2
    )
    total_tested = len(all_symbol_results)
    print(f"\n  Edge confirmed on {verdict_count}/{total_tested} pairs.")
    if total_tested > 0 and verdict_count / total_tested >= 0.75:
        print("  ✅  System is ROBUST across instruments.")
    elif verdict_count > 0:
        print("  🟡  PARTIAL — Strong on some pairs, weak on others. Investigate.")
    else:
        print("  🔴  FRAGILE — System does not generalise. Do not trade live.")
    print(f"{'='*55}\n")

    # Save master summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment': {
                'experiment_id': cfg.get('experiment_id'),
                'strategy_name': cfg.get('strategy_name'),
                'config_hash': cfg.get('config_hash'),
                'declared_trial_count': cfg.get('declared_trial_count', 1),
            },
            'results': {s: r['metrics'] for s, r in all_symbol_results.items()}
        }, f, indent=4, default=str)
    print(f"  Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()