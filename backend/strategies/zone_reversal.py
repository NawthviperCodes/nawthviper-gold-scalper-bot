"""Canonical adapter for the existing v2 zone-reversal decision rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from trading_core import (
    Side,
    SignalEvent,
    StrategySnapshot,
    ZoneEvent,
    ZoneType,
    stable_event_id,
)


DecisionFunction = Callable[..., Tuple[list, list]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ZoneReversalConfig:
    strategy_id: str = "zone_reversal_phase4"
    point: float
    tp_ratio: float = 2.0
    pattern_bars: int = 5
    strategy_active: Optional[bool] = True
    emit_legacy_telemetry: bool = True
    thresholds: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if self.point <= 0 or self.tp_ratio <= 0:
            raise ValueError("point and tp_ratio must be positive")
        if self.pattern_bars < 2:
            raise ValueError("pattern_bars must be at least two")

    @classmethod
    def from_thresholds(cls, *, thresholds: Mapping[str, float], **kwargs):
        normalized = tuple(sorted((str(key), float(value)) for key, value in thresholds.items()))
        return cls(thresholds=normalized, **kwargs)

    def threshold_dict(self) -> dict:
        return dict(self.thresholds)


def _bars_frame(snapshot: StrategySnapshot, timeframe: str) -> pd.DataFrame:
    rows = [
        {
            "time": bar.opened_at,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "tick_volume": bar.tick_volume,
        }
        for bar in snapshot.bars(timeframe)
    ]
    return pd.DataFrame(rows)


def _trend(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 51:
        return None
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    return "uptrend" if float(df["close"].iloc[-1]) > float(sma50) else "downtrend"


def _htf_bias(df: pd.DataFrame) -> str:
    if len(df) < 200:
        return "NEUTRAL"
    fast = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    slow = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    if fast > slow:
        return "UP"
    if fast < slow:
        return "DOWN"
    return "NEUTRAL"


def wilder_atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window:
        raise ValueError(f"ATR requires at least {window} closed bars")
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    values = true_range.to_numpy(dtype=float)
    atr = float(np.mean(values[:window]))
    for value in values[window:]:
        atr = ((window - 1) * atr + float(value)) / window
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ATR calculation produced a non-positive value")
    return atr


def _legacy_zone(zone: ZoneEvent) -> dict:
    price = zone.bottom if zone.zone_type is ZoneType.DEMAND else zone.top
    origin_low = getattr(zone, "displacement_origin_low", None)
    origin_high = getattr(zone, "displacement_origin_high", None)
    return {
        "type": zone.zone_type.value,
        "price": float(price),
        "bottom": float(zone.bottom),
        "top": float(zone.top),
        "quality_bucket": zone.quality,
        "freshness_score": float(zone.freshness),
        "touches": int(zone.touch_count),
        "zone_id": zone.event_id,
        # ZoneEvent currently carries the displacement candle's boundaries as
        # bottom/top. Explicit attributes take precedence when introduced.
        "displacement_origin_low": float(
            zone.bottom if origin_low is None else origin_low
        ),
        "displacement_origin_high": float(
            zone.top if origin_high is None else origin_high
        ),
    }


class ZoneReversalStrategy:
    """Translate canonical snapshots to/from the existing decision function."""

    def __init__(self, config: ZoneReversalConfig, decision_fn: Optional[DecisionFunction] = None):
        self.config = config
        self.strategy_id = config.strategy_id
        self._decision_fn = decision_fn

    def _resolve_decision_fn(self) -> DecisionFunction:
        if self._decision_fn is None:
            from trade_decision_engine import run_trade_decision_engine

            self._decision_fn = run_trade_decision_engine
        return self._decision_fn

    def evaluate(self, snapshot: StrategySnapshot) -> Tuple[SignalEvent, ...]:
        if snapshot.quote is None:
            raise ValueError("zone-reversal strategy requires a causal market quote")

        h4 = _bars_frame(snapshot, "H4")
        h1 = _bars_frame(snapshot, "H1")
        m5 = _bars_frame(snapshot, "M5")
        if len(h4) < 200 or len(h1) < 80 or len(m5) < max(51, self.config.pattern_bars):
            return ()

        bias = _htf_bias(h4)

        demand_events = tuple(z for z in snapshot.active_zones if z.zone_type is ZoneType.DEMAND)
        supply_events = tuple(z for z in snapshot.active_zones if z.zone_type is ZoneType.SUPPLY)
        demand = [_legacy_zone(z) for z in demand_events]
        supply = [_legacy_zone(z) for z in supply_events]
        if not demand and not supply:
            return ()

        positions = {}
        if snapshot.positions:
            position = snapshot.positions[0]
            positions[snapshot.symbol] = {
                "side": position.side.value,
                "ticket": position.position_id,
            }

        atr = wilder_atr(m5)
        h1_atr = wilder_atr(h1)
        decision_fn = self._resolve_decision_fn()
        legacy_signals, _ = decision_fn(
            symbol=snapshot.symbol,
            point=self.config.point,
            current_price=float(snapshot.quote.bid),
            trend=_trend(h1),
            demand_zones=demand,
            supply_zones=supply,
            fast_demand_zones=[],
            fast_supply_zones=[],
            m1_candles_for_crt=None,
            m5_candles_for_patterns=m5.iloc[-self.config.pattern_bars :],
            active_trades=positions,
            zone_touch_counts={},
            SL_BUFFER=0,
            TP_RATIO=self.config.tp_ratio,
            CHECK_RANGE=max(atr, 50 * self.config.point),
            LOT_SIZE=0.0,
            MAGIC=0,
            strategy_mode=self.strategy_id,
            atr=atr,
            htf_atr=h1_atr,
            m5_context={"trend": _trend(m5)},
            htf_high=float(h1["high"].max()),
            htf_low=float(h1["low"].min()),
            last_closed_h1=h1.iloc[-1],
            htf_bias=bias,
            thresholds=self.config.threshold_dict(),
            strategy_active_override=self.config.strategy_active,
            emit_telemetry=self.config.emit_legacy_telemetry,
            spread=max(0.0, float(snapshot.quote.ask) - float(snapshot.quote.bid)),
        )

        all_zones = demand_events + supply_events
        canonical = []
        for legacy in legacy_signals:
            side = Side(str(legacy["side"]).lower())
            expected_type = ZoneType.DEMAND if side is Side.BUY else ZoneType.SUPPLY
            candidates = [zone for zone in all_zones if zone.zone_type is expected_type]
            if not candidates:
                raise ValueError("legacy signal cannot be mapped to a canonical zone")
            legacy_mid = float(legacy.get("zone", {}).get("mid"))
            matched = min(
                candidates,
                key=lambda zone: abs(
                    (zone.bottom if zone.zone_type is ZoneType.DEMAND else zone.top) - legacy_mid
                ),
            )
            reclaim_id = stable_event_id(
                "reclaim", snapshot.symbol, matched.event_id, side, snapshot.as_of
            )
            entry = float(legacy["entry"])
            stop = float(legacy["sl"])
            target = float(legacy["tp"])
            event_id = stable_event_id(
                "signal",
                snapshot.symbol,
                self.strategy_id,
                matched.event_id,
                side,
                snapshot.as_of,
                entry,
                stop,
                target,
            )
            canonical.append(
                SignalEvent(
                    event_id=event_id,
                    symbol=snapshot.symbol,
                    occurred_at=snapshot.as_of,
                    known_at=snapshot.as_of,
                    parent_event_id=reclaim_id,
                    metadata=(("legacy_reason", str(legacy.get("reason", ""))),),
                    strategy_id=self.strategy_id,
                    side=side,
                    zone_id=matched.event_id,
                    reclaim_event_id=reclaim_id,
                    signal_at=snapshot.as_of,
                    entry_reference=entry,
                    stop_loss=stop,
                    take_profit=target,
                    score=float(legacy.get("confidence", 0.0)),
                )
            )

        return tuple(canonical)
