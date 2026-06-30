"""One strategy boundary for replay, paper, and live adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Protocol, Tuple, runtime_checkable

from .events import MarketBar, MarketQuote, PositionState, SignalEvent, ZoneEvent


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True, kw_only=True)
class TimeframeSeries:
    timeframe: str
    bars: Tuple[MarketBar, ...]

    def __post_init__(self) -> None:
        if not self.timeframe.strip():
            raise ValueError("timeframe must be non-empty")
        if any(bar.timeframe != self.timeframe for bar in self.bars):
            raise ValueError("all bars must match the series timeframe")
        if any(a.opened_at >= b.opened_at for a, b in zip(self.bars, self.bars[1:])):
            raise ValueError("bars must be strictly ordered by opened_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class StrategySnapshot:
    symbol: str
    as_of: datetime
    series: Tuple[TimeframeSeries, ...]
    active_zones: Tuple[ZoneEvent, ...] = field(default_factory=tuple)
    quote: Optional[MarketQuote] = None
    positions: Tuple[PositionState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        _require_utc(self.as_of, "as_of")
        timeframe_names = [item.timeframe for item in self.series]
        if len(timeframe_names) != len(set(timeframe_names)):
            raise ValueError("snapshot cannot contain duplicate timeframes")
        for item in self.series:
            for bar in item.bars:
                if bar.symbol != self.symbol:
                    raise ValueError("bar symbol does not match snapshot symbol")
                if bar.known_at > self.as_of:
                    raise ValueError("snapshot contains a bar unknown at as_of")
        for zone in self.active_zones:
            if zone.symbol != self.symbol:
                raise ValueError("zone symbol does not match snapshot symbol")
            if zone.known_at > self.as_of:
                raise ValueError("snapshot contains a zone unknown at as_of")
        if self.quote is not None:
            if self.quote.symbol != self.symbol:
                raise ValueError("quote symbol does not match snapshot symbol")
            if self.quote.known_at > self.as_of:
                raise ValueError("snapshot contains a quote unknown at as_of")
        for position in self.positions:
            if position.symbol != self.symbol:
                raise ValueError("position symbol does not match snapshot symbol")
            if position.known_at > self.as_of:
                raise ValueError("snapshot contains a position unknown at as_of")

    def bars(self, timeframe: str) -> Tuple[MarketBar, ...]:
        for item in self.series:
            if item.timeframe == timeframe:
                return item.bars
        return ()


@runtime_checkable
class TradingStrategy(Protocol):
    strategy_id: str

    def evaluate(self, snapshot: StrategySnapshot) -> Tuple[SignalEvent, ...]:
        """Return deterministic signals using only the supplied snapshot."""
        ...


class StrategyRunner:
    """Enforce the same causal output contract in every environment."""

    def run(self, strategy: TradingStrategy, snapshot: StrategySnapshot) -> Tuple[SignalEvent, ...]:
        signals = tuple(strategy.evaluate(snapshot))
        seen = set()
        for signal in signals:
            if signal.event_id in seen:
                raise ValueError("strategy emitted duplicate event IDs")
            seen.add(signal.event_id)
            if signal.symbol != snapshot.symbol:
                raise ValueError("signal symbol does not match snapshot")
            if signal.strategy_id != strategy.strategy_id:
                raise ValueError("signal strategy_id does not match strategy")
            if signal.known_at != snapshot.as_of:
                raise ValueError("signals must be created exactly at snapshot.as_of")
        return signals
