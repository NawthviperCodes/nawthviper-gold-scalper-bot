"""Feature-gated boundary between legacy calls and canonical strategy events."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite
from typing import Callable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from trading_core import (
    MarketBar,
    MarketQuote,
    PositionState,
    Side,
    StrategyRunner,
    StrategySnapshot,
    TimeframeSeries,
    ZoneEvent,
    ZoneType,
    stable_event_id,
)

from .zone_reversal import ZoneReversalConfig, ZoneReversalStrategy


PIPELINE_MODES = frozenset({"legacy", "shadow", "canonical"})
_TIMEFRAME_LENGTHS = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
}


@dataclass(frozen=True, slots=True)
class ParityReport:
    symbol: str
    as_of: datetime
    matched: bool
    legacy_signals: Tuple[tuple, ...]
    canonical_signals: Tuple[tuple, ...]
    canonical_error: Optional[str] = None


def normalize_pipeline_mode(value: object) -> str:
    mode = str(value or "legacy").strip().lower()
    if mode not in PIPELINE_MODES:
        allowed = ", ".join(sorted(PIPELINE_MODES))
        raise ValueError(f"STRATEGY_PIPELINE_MODE must be one of: {allowed}")
    return mode


def timeframe_close_delta(timeframe: str) -> timedelta:
    try:
        return _TIMEFRAME_LENGTHS[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported canonical timeframe: {timeframe}") from exc


def _utc(value: object) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def quote_rejection_reason(
    *,
    bid: object,
    ask: object,
    occurred_at: object,
    observed_at: object,
    max_age_seconds: float,
    future_tolerance_seconds: float = 5.0,
) -> Optional[str]:
    """Return a stable fail-closed reason for an unusable broker quote."""
    if max_age_seconds <= 0 or future_tolerance_seconds < 0:
        raise ValueError("quote age limits must be positive")
    if occurred_at is None or observed_at is None:
        return "invalid"
    try:
        bid_value = float(bid)
        ask_value = float(ask)
    except (TypeError, ValueError):
        return "invalid"
    if (
        not isfinite(bid_value)
        or not isfinite(ask_value)
        or bid_value <= 0
        or ask_value <= 0
        or ask_value < bid_value
    ):
        return "invalid"
    try:
        age_seconds = (_utc(observed_at) - _utc(occurred_at)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return "invalid"
    if age_seconds < -future_tolerance_seconds:
        return "future"
    if age_seconds > max_age_seconds:
        return "stale"
    return None


def closed_bars_frame(
    frame: pd.DataFrame, *, timeframe: str, as_of: object, tail: Optional[int] = None
) -> pd.DataFrame:
    """Select bars whose close is knowable at ``as_of`` using open timestamps."""
    if "time" not in frame.columns:
        raise ValueError("market dataframe requires a time column")
    opened = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True))
    selection = closed_bars_slice(opened, timeframe=timeframe, as_of=as_of, tail=tail)
    return frame.iloc[selection].copy()


def closed_bars_slice(
    opened_at: pd.DatetimeIndex,
    *,
    timeframe: str,
    as_of: object,
    tail: Optional[int] = None,
) -> slice:
    """Return an O(log n) positional slice of bars closed by ``as_of``."""
    if not isinstance(opened_at, pd.DatetimeIndex):
        opened_at = pd.DatetimeIndex(pd.to_datetime(opened_at, utc=True))
    if not opened_at.is_monotonic_increasing:
        raise ValueError("opened_at must be sorted in ascending order")
    if tail is not None and tail <= 0:
        raise ValueError("tail must be positive")
    cutoff = pd.Timestamp(_utc(as_of)) - timeframe_close_delta(timeframe)
    end = int(opened_at.searchsorted(cutoff, side="right"))
    start = max(0, end - tail) if tail is not None else 0
    return slice(start, end)


def dataframe_to_series(
    *, symbol: str, timeframe: str, frame: pd.DataFrame, as_of: datetime
) -> TimeframeSeries:
    """Convert only fully closed dataframe rows into canonical market bars."""
    if "time" not in frame.columns:
        raise ValueError("market dataframe requires a time column")

    duration = timeframe_close_delta(timeframe)
    bars = []
    for row in frame.itertuples(index=False):
        opened_at = _utc(row.time)
        closed_at = opened_at + duration
        if closed_at > as_of:
            continue
        bars.append(
            MarketBar(
                symbol=symbol,
                timeframe=timeframe,
                opened_at=opened_at,
                closed_at=closed_at,
                known_at=closed_at,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                tick_volume=float(getattr(row, "tick_volume", 0.0) or 0.0),
            )
        )
    return TimeframeSeries(timeframe=timeframe, bars=tuple(bars))


def legacy_zones_to_events(
    *, symbol: str, zones: Sequence[Mapping], zone_type: ZoneType, as_of: datetime
) -> Tuple[ZoneEvent, ...]:
    events = []
    for zone in zones:
        created_value = zone.get("created_at")
        if created_value is None:
            created_value = zone.get("time")
        if created_value is None:
            raise ValueError("legacy zone requires created_at or time")
        created_at = _utc(created_value)
        activated_at = _utc(zone.get("activated_at", zone.get("detected_at", created_at)))
        detected_at = _utc(zone.get("detected_at", activated_at))
        if activated_at > as_of:
            continue
        detected_at = min(max(detected_at, created_at), activated_at)
        bottom = float(zone["bottom"])
        top = float(zone["top"])
        event_id = str(zone.get("zone_id") or stable_event_id(
            "zone", symbol, zone_type, created_at, activated_at, bottom, top
        ))
        events.append(
            ZoneEvent(
                event_id=event_id,
                symbol=symbol,
                occurred_at=created_at,
                known_at=activated_at,
                zone_type=zone_type,
                created_at=created_at,
                detected_at=detected_at,
                activated_at=activated_at,
                bottom=bottom,
                top=top,
                quality=str(zone.get("quality_bucket", "C")),
                freshness=min(1.0, max(0.0, float(zone.get("freshness_score", 0.0)))),
                touch_count=max(0, int(zone.get("touches", 0) or 0)),
            )
        )
    return tuple(events)


def _position_states(
    *, symbol: str, active_trades: Mapping, as_of: datetime, fallback_price: float
) -> Tuple[PositionState, ...]:
    raw = active_trades.get(symbol) if active_trades else None
    if not raw:
        return ()
    opened_at = _utc(raw.get("opened_at", as_of))
    return (
        PositionState(
            position_id=str(raw.get("ticket", "legacy-position")),
            symbol=symbol,
            side=Side(str(raw["side"]).lower()),
            volume=float(raw.get("volume", 1.0) or 1.0),
            entry_price=float(raw.get("entry_price", fallback_price) or fallback_price),
            opened_at=min(opened_at, as_of),
            known_at=as_of,
            stop_loss=float(raw["stop_loss"]) if raw.get("stop_loss") else None,
            take_profit=float(raw["take_profit"]) if raw.get("take_profit") else None,
        ),
    )


def build_zone_reversal_snapshot(
    *,
    symbol: str,
    as_of: datetime,
    bid: float,
    ask: float,
    h4: pd.DataFrame,
    h1: pd.DataFrame,
    m5: pd.DataFrame,
    demand_zones: Sequence[Mapping],
    supply_zones: Sequence[Mapping],
    active_trades: Optional[Mapping] = None,
) -> StrategySnapshot:
    as_of = _utc(as_of)
    quote = MarketQuote(symbol=symbol, bid=float(bid), ask=float(ask), occurred_at=as_of, known_at=as_of)
    zones = (
        legacy_zones_to_events(symbol=symbol, zones=demand_zones, zone_type=ZoneType.DEMAND, as_of=as_of)
        + legacy_zones_to_events(symbol=symbol, zones=supply_zones, zone_type=ZoneType.SUPPLY, as_of=as_of)
    )
    return StrategySnapshot(
        symbol=symbol,
        as_of=as_of,
        series=(
            dataframe_to_series(symbol=symbol, timeframe="H4", frame=h4, as_of=as_of),
            dataframe_to_series(symbol=symbol, timeframe="H1", frame=h1, as_of=as_of),
            dataframe_to_series(symbol=symbol, timeframe="M5", frame=m5, as_of=as_of),
        ),
        active_zones=zones,
        quote=quote,
        positions=_position_states(
            symbol=symbol,
            active_trades=active_trades or {},
            as_of=as_of,
            fallback_price=float(bid),
        ),
    )


def canonical_signals_to_legacy(snapshot: StrategySnapshot, signals) -> list:
    zones = {zone.event_id: zone for zone in snapshot.active_zones}
    output = []
    for signal in signals:
        zone = zones[signal.zone_id]
        reason = dict(signal.metadata).get("legacy_reason", "")
        price = zone.bottom if zone.zone_type is ZoneType.DEMAND else zone.top
        output.append({
            "side": signal.side.value,
            "entry": signal.entry_reference,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "zone": {"mid": float(price)},
            "strategy": signal.strategy_id,
            "reason": reason,
            "confidence": signal.score,
        })
    output.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
    return output


def _signature(signals: Sequence[Mapping]) -> Tuple[tuple, ...]:
    return tuple(sorted((
        str(item.get("side", "")),
        round(float(item.get("entry", 0.0)), 10),
        round(float(item.get("sl", 0.0)), 10),
        round(float(item.get("tp", 0.0)), 10),
        round(float(item.get("confidence", 0.0)), 4),
        round(float(item.get("zone", {}).get("mid", 0.0)), 10),
        str(item.get("strategy", "")),
        str(item.get("reason", "")),
    ) for item in signals))


def run_zone_reversal_pipeline(
    *,
    mode: str,
    legacy_kwargs: Mapping,
    snapshot: Optional[StrategySnapshot] = None,
    config: Optional[ZoneReversalConfig] = None,
    decision_fn: Optional[Callable] = None,
    parity_sink: Optional[Callable[[ParityReport], None]] = None,
):
    """Run legacy, shadow, or canonical logic while retaining the legacy return contract."""
    mode = normalize_pipeline_mode(mode)
    if decision_fn is None:
        from trade_decision_engine import run_trade_decision_engine

        decision_fn = run_trade_decision_engine

    if mode == "legacy":
        return decision_fn(**dict(legacy_kwargs))
    if snapshot is None or config is None:
        raise ValueError("canonical and shadow modes require a snapshot and strategy config")

    def canonical_run():
        strategy_config = config
        if mode == "shadow" and strategy_config.emit_legacy_telemetry:
            strategy_config = replace(strategy_config, emit_legacy_telemetry=False)
        strategy = ZoneReversalStrategy(strategy_config, decision_fn=decision_fn)
        events = StrategyRunner().run(strategy, snapshot)
        return canonical_signals_to_legacy(snapshot, events)

    if mode == "canonical":
        return canonical_run(), []

    legacy_signals, legacy_rejections = decision_fn(**dict(legacy_kwargs))
    canonical_error = None
    try:
        canonical_signals = canonical_run()
    except Exception as exc:  # shadow mode must never alter production decisions
        canonical_signals = []
        canonical_error = f"{type(exc).__name__}: {exc}"
    legacy_signature = _signature(legacy_signals)
    canonical_signature = _signature(canonical_signals)
    if parity_sink is not None:
        parity_sink(ParityReport(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            matched=canonical_error is None and legacy_signature == canonical_signature,
            legacy_signals=legacy_signature,
            canonical_signals=canonical_signature,
            canonical_error=canonical_error,
        ))
    return legacy_signals, legacy_rejections
