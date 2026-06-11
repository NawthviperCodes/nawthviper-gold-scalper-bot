# ======================================================
# === trade_decision_engine_v2.py ======================
# ======================================================
#
# Simplified v2 research engine
#
# Built only from the Phase 1–4 findings:
# - Phase 1C zones
# - reversal-first
# - tap/deep touches only
# - no mandatory candlestick gate
# - doji/pin bars are optional bonuses
# - stop = beyond sweep wick + 0.15 ATR
# - TP = fixed 2R
#
# Signature compatibility is preserved so existing live/backtest code can call
# this file without structural changes.

from __future__ import annotations

from datetime import datetime
import math
from typing import List, Optional, Tuple

from performance_tracker import is_strategy_active
from candlestick_patterns import (
    is_bullish_pin_bar,
    is_bearish_pin_bar,
)

REJECTION_STATS = {
    "Weak Quality Zone": 0,
    "Too Many Touches": 0,
    "No Recent Touch": 0,
    "Full Touch Skip": 0,
    "No Reclaim Confirmation": 0,
    "Low RR": 0,
    "Trade Conflict": 0,
    "Inactive Strategy": 0,
}

rejected_signals_log: List[dict] = []


def log_rejection(reason, zone_type, zone_price, strategy, trend):
    rejected_signals_log.append({
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "zone_type": zone_type,
        "zone_price": zone_price,
        "strategy": strategy,
        "trend": trend,
    })


def _safe_atr(atr: Optional[float], current_price: float) -> float:
    if atr is not None:
        try:
            atr = float(atr)
            if not math.isnan(atr) and atr > 0:
                return atr
        except Exception:
            pass
    return max(float(current_price) * 0.0005, 1e-9)


def _is_doji(candle) -> bool:
    rng = float(candle.high - candle.low)
    if rng <= 0:
        return False
    body = abs(float(candle.close - candle.open))
    return body <= rng * 0.10


def _pattern_bonus(m5_df, side: str) -> Tuple[float, str]:
    if m5_df is None or len(m5_df) < 1:
        return 0.0, "no_pattern"

    c = m5_df.iloc[-1]

    if _is_doji(c):
        return 0.06, "doji"

    if side == "buy" and is_bullish_pin_bar(c.open, c.high, c.low, c.close):
        return 0.05, "bullish_pin_bar"

    if side == "sell" and is_bearish_pin_bar(c.open, c.high, c.low, c.close):
        return 0.05, "bearish_pin_bar"

    return 0.0, "no_pattern"


def _infer_quality(zone: dict) -> str:
    q = str(zone.get("quality_bucket", "")).upper().strip()
    if q in {"A", "B", "C"}:
        return q

    width_atr = float(zone.get("width_atr_ratio", 0.0) or 0.0)
    dep_atr = float(zone.get("departure_strength_atr", 0.0) or 0.0)
    touches = int(zone.get("touches", 0) or 0)

    if dep_atr >= 2.5 and 0.20 <= width_atr <= 0.90 and touches <= 1:
        return "A"
    if dep_atr >= 1.6 and 0.15 <= width_atr <= 1.15 and touches <= 2:
        return "B"
    return "C"


def _freshness_class(zone: dict, max_touch_allowed: int) -> str:
    touches = int(zone.get("touches", 0) or 0)
    if touches <= 0:
        return "fresh"
    if touches == 1:
        return "first_retest"
    if touches == 2:
        return "second_retest"
    if touches <= max_touch_allowed:
        return "late_retest"
    return "overused"


def _classify_touch_depth(zone_bottom: float, zone_top: float, low: float, high: float, side: str) -> str:
    width = max(zone_top - zone_bottom, 1e-9)
    if side == "buy":
        penetration = max(0.0, zone_top - low)
    else:
        penetration = max(0.0, high - zone_bottom)
    ratio = penetration / width
    if ratio <= 0.20:
        return "tap"
    if ratio <= 0.75:
        return "deep"
    return "full"


def _recent_touch_context(m5_df, zone_bottom: float, zone_top: float, side: str, lookback: int = 15):
    if m5_df is None or len(m5_df) == 0:
        return None

    recent = m5_df.tail(lookback).copy()
    touched_mask = (recent["high"] >= zone_bottom) & (recent["low"] <= zone_top)
    touched = recent[touched_mask]
    if touched.empty:
        return None

    touch_low = float(touched["low"].min())
    touch_high = float(touched["high"].max())
    depth = _classify_touch_depth(zone_bottom, zone_top, touch_low, touch_high, side)

    return {
        "touch_count_recent": int(len(touched)),
        "touch_low": touch_low,
        "touch_high": touch_high,
        "touch_depth": depth,
        "sweep_wick": touch_low if side == "buy" else touch_high,
        "last_touch_candle": touched.iloc[-1],
        "recent_df": recent,
        "touched_df": touched,
    }


def _simple_reclaim_confirmation(m5_df, zone_bottom: float, zone_top: float, side: str) -> Tuple[bool, str]:
    if m5_df is None or len(m5_df) < 2:
        return False, "not_enough_m5"

    c_prev = m5_df.iloc[-2]
    c_last = m5_df.iloc[-1]
    zone_mid = (zone_top + zone_bottom) / 2.0

    if side == "buy":
        ok = (
            c_last.close > c_last.open and
            c_last.close > zone_mid and
            c_last.close > c_prev.close
        )
        return ok, ("buy_reclaim" if ok else "no_buy_reclaim")

    ok = (
        c_last.close < c_last.open and
        c_last.close < zone_mid and
        c_last.close < c_prev.close
    )
    return ok, ("sell_reclaim" if ok else "no_sell_reclaim")


def _trend_bonus(m5_context, side: str) -> float:
    if not m5_context:
        return 0.0
    m5_trend = m5_context.get("trend")
    if side == "buy" and m5_trend == "uptrend":
        return 0.04
    if side == "sell" and m5_trend == "downtrend":
        return 0.04
    if side == "buy" and m5_trend == "downtrend":
        return -0.02
    if side == "sell" and m5_trend == "uptrend":
        return -0.02
    return 0.0


def _htf_soft_bonus(htf_bias: str, side: str) -> float:
    if side == "buy" and htf_bias == "UP":
        return 0.03
    if side == "sell" and htf_bias == "DOWN":
        return 0.03
    if side == "buy" and htf_bias == "DOWN":
        return -0.02
    if side == "sell" and htf_bias == "UP":
        return -0.02
    return 0.0


def _build_signal(
    symbol: str,
    side: str,
    current_price: float,
    zone: dict,
    touch_ctx: dict,
    atr_val: float,
    tp_ratio: float,
    strategy_name: str,
    reason: str,
    confidence: float,
):
    entry_price = float(current_price)
    wick = float(touch_ctx["sweep_wick"])
    stop_buffer = atr_val * 0.15

    if side == "buy":
        sl = wick - stop_buffer
        risk = entry_price - sl
        if risk <= 0:
            return None
        tp = entry_price + risk * tp_ratio
    else:
        sl = wick + stop_buffer
        risk = sl - entry_price
        if risk <= 0:
            return None
        tp = entry_price - risk * tp_ratio

    return {
        "side": side,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "zone": {"mid": float(zone.get("price", (zone["top"] + zone["bottom"]) / 2.0))},
        "strategy": strategy_name,
        "reason": reason,
        "confidence": round(min(max(confidence, 0.0), 0.95), 4),
    }


def run_trade_decision_engine(
    symbol, point, current_price, trend, demand_zones, supply_zones,
    fast_demand_zones=[], fast_supply_zones=[],
    m1_candles_for_crt=None, m5_candles_for_patterns=None,
    active_trades=None, zone_touch_counts=None,
    SL_BUFFER=0, TP_RATIO=1.5, CHECK_RANGE=0, LOT_SIZE=0.01, MAGIC=0,
    strategy_mode="zone_reversal_v2",
    macd=None, macd_signal=None, rsi=None, vwap=None,
    atr=None, htf_atr=None,
    m5_context=None, htf_high=None, htf_low=None, last_closed_h1=None,
    fibo_zone=None, bollinger_bands=None,
    htf_bias="NEUTRAL", thresholds={}
):
    signals = []

    if m5_candles_for_patterns is None or len(m5_candles_for_patterns) < 2:
        return [], []

    atr_val = _safe_atr(atr, current_price)
    max_touch_allowed = int(thresholds.get("MAX_TOUCH_ALLOWED", 2))
    rr_min = float(thresholds.get("MIN_RR_FILTER", 1.30))

    for zone_type, zones in [("demand", demand_zones), ("supply", supply_zones)]:
        side = "buy" if zone_type == "demand" else "sell"

        for zone in zones:
            zone_bottom = float(zone.get("bottom", 0.0))
            zone_top = float(zone.get("top", 0.0))
            if zone_top <= zone_bottom:
                continue

            quality = _infer_quality(zone)
            if quality not in {"A", "B"}:
                REJECTION_STATS["Weak Quality Zone"] += 1
                log_rejection("Weak Quality Zone", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            zone_touches = int(zone.get("touches", 0) or 0)
            if zone_touches > max_touch_allowed:
                REJECTION_STATS["Too Many Touches"] += 1
                log_rejection("Too Many Touches", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            freshness_class = _freshness_class(zone, max_touch_allowed)
            if freshness_class == "overused":
                REJECTION_STATS["Too Many Touches"] += 1
                log_rejection("Too Many Touches", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            touch_ctx = _recent_touch_context(
                m5_candles_for_patterns,
                zone_bottom,
                zone_top,
                side,
                lookback=15,
            )
            if touch_ctx is None:
                REJECTION_STATS["No Recent Touch"] += 1
                log_rejection("No Recent Touch", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            if touch_ctx["touch_depth"] == "full":
                REJECTION_STATS["Full Touch Skip"] += 1
                log_rejection("Full Touch Skip", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            confirmed, confirm_reason = _simple_reclaim_confirmation(
                m5_candles_for_patterns,
                zone_bottom,
                zone_top,
                side,
            )
            if not confirmed:
                REJECTION_STATS["No Reclaim Confirmation"] += 1
                log_rejection("No Reclaim Confirmation", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            confidence = 0.56 if quality == "B" else 0.64

            if freshness_class == "fresh":
                confidence += 0.05
            elif freshness_class == "first_retest":
                confidence += 0.04
            elif freshness_class == "second_retest":
                confidence += 0.01

            if touch_ctx["touch_depth"] == "tap":
                confidence += 0.06
            elif touch_ctx["touch_depth"] == "deep":
                confidence += 0.03

            confidence += _trend_bonus(m5_context, side)
            confidence += _htf_soft_bonus(htf_bias, side)

            pattern_bonus, pattern_name = _pattern_bonus(m5_candles_for_patterns, side)
            confidence += pattern_bonus

            if active_trades and isinstance(active_trades, dict) and symbol in active_trades:
                if active_trades[symbol].get("side") != side:
                    REJECTION_STATS["Trade Conflict"] += 1
                    log_rejection("Trade Conflict", zone_type, zone.get("price"), strategy_mode, trend)
                    continue

            if not is_strategy_active(strategy_mode):
                REJECTION_STATS["Inactive Strategy"] += 1
                log_rejection("Inactive Strategy", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            reason = confirm_reason if pattern_name == "no_pattern" else f"{confirm_reason}+{pattern_name}"
            sig = _build_signal(
                symbol=symbol,
                side=side,
                current_price=current_price,
                zone=zone,
                touch_ctx=touch_ctx,
                atr_val=atr_val,
                tp_ratio=float(TP_RATIO),
                strategy_name=strategy_mode,
                reason=reason,
                confidence=confidence,
            )
            if sig is None:
                continue

            risk = abs(sig["entry"] - sig["sl"])
            reward = abs(sig["tp"] - sig["entry"])
            rr = reward / risk if risk > 0 else 0.0
            if rr < rr_min:
                REJECTION_STATS["Low RR"] += 1
                log_rejection("Low RR", zone_type, zone.get("price"), strategy_mode, trend)
                continue

            signals.append(sig)

    signals.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return signals, []


def format_confidence_label(score):
    if score >= 0.80:
        return "🔥 HIGH"
    if score >= 0.65:
        return "✅ GOOD"
    if score >= 0.50:
        return "🟡 VALID"
    return "⚠️ LOW"


def print_rejection_summary():
    print("\n===== REJECTION SUMMARY (v2) =====")
    total = sum(REJECTION_STATS.values())
    for k, v in REJECTION_STATS.items():
        pct = (v / total * 100) if total > 0 else 0.0
        print(f"{k}: {v} ({pct:.1f}%)")
    print("==================================\n")
