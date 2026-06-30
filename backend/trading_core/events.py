"""Canonical immutable events shared by research, replay, paper, and live paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Optional, Tuple
from uuid import NAMESPACE_URL, uuid5


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ZoneType(str, Enum):
    DEMAND = "demand"
    SUPPLY = "supply"


class TouchDepth(str, Enum):
    TAP = "tap"
    DEEP = "deep"
    FULL = "full"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be normalized to UTC")


def _require_price(value: float, name: str) -> None:
    if not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def stable_event_id(kind: str, symbol: str, *identity_parts: object) -> str:
    """Return a deterministic ID for replay/live idempotency.

    Identity inputs must contain only information available when the event is
    created.  Replaying the same causal stream therefore produces the same ID.
    """
    _require_text(kind, "kind")
    _require_text(symbol, "symbol")
    normalized = []
    for value in identity_parts:
        if isinstance(value, datetime):
            _require_utc(value, "identity datetime")
            normalized.append(value.isoformat(timespec="microseconds"))
        elif isinstance(value, Enum):
            normalized.append(str(value.value))
        else:
            normalized.append(str(value))
    payload = "|".join([kind.strip(), symbol.strip().upper(), *normalized])
    return str(uuid5(NAMESPACE_URL, payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEvent:
    event_id: str
    symbol: str
    occurred_at: datetime
    known_at: datetime
    parent_event_id: Optional[str] = None
    metadata: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.symbol, "symbol")
        _require_utc(self.occurred_at, "occurred_at")
        _require_utc(self.known_at, "known_at")
        if self.known_at < self.occurred_at:
            raise ValueError("known_at cannot precede occurred_at")
        if self.parent_event_id is not None:
            _require_text(self.parent_event_id, "parent_event_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketBar:
    symbol: str
    timeframe: str
    opened_at: datetime
    closed_at: datetime
    known_at: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")
        _require_text(self.timeframe, "timeframe")
        for name in ("opened_at", "closed_at", "known_at"):
            _require_utc(getattr(self, name), name)
        if not self.opened_at < self.closed_at:
            raise ValueError("opened_at must precede closed_at")
        if self.known_at < self.closed_at:
            raise ValueError("a bar cannot be known before it closes")
        for name in ("open", "high", "low", "close"):
            _require_price(getattr(self, name), name)
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high violates OHLC ordering")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low violates OHLC ordering")
        if not isfinite(float(self.tick_volume)) or self.tick_volume < 0:
            raise ValueError("tick_volume must be finite and non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketQuote:
    symbol: str
    bid: float
    ask: float
    occurred_at: datetime
    known_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")
        _require_price(self.bid, "bid")
        _require_price(self.ask, "ask")
        _require_utc(self.occurred_at, "occurred_at")
        _require_utc(self.known_at, "known_at")
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        if self.known_at < self.occurred_at:
            raise ValueError("quote known_at cannot precede occurred_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionState:
    position_id: str
    symbol: str
    side: Side
    volume: float
    entry_price: float
    opened_at: datetime
    known_at: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self) -> None:
        _require_text(self.position_id, "position_id")
        _require_text(self.symbol, "symbol")
        if not isfinite(float(self.volume)) or self.volume <= 0:
            raise ValueError("volume must be finite and positive")
        _require_price(self.entry_price, "entry_price")
        _require_utc(self.opened_at, "opened_at")
        _require_utc(self.known_at, "known_at")
        if self.known_at < self.opened_at:
            raise ValueError("position cannot be known before it opens")
        for name in ("stop_loss", "take_profit"):
            value = getattr(self, name)
            if value is not None:
                _require_price(value, name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZoneEvent(DomainEvent):
    zone_type: ZoneType
    created_at: datetime
    detected_at: datetime
    activated_at: datetime
    bottom: float
    top: float
    quality: str
    freshness: float
    touch_count: int = 0

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        for name in ("created_at", "detected_at", "activated_at"):
            _require_utc(getattr(self, name), name)
        if not self.created_at <= self.detected_at <= self.activated_at:
            raise ValueError("zone timestamps must satisfy created <= detected <= activated")
        if self.occurred_at != self.created_at or self.known_at != self.activated_at:
            raise ValueError("zone occurred_at/known_at must equal created_at/activated_at")
        _require_price(self.bottom, "bottom")
        _require_price(self.top, "top")
        if self.top <= self.bottom:
            raise ValueError("zone top must exceed bottom")
        _require_text(self.quality, "quality")
        if not 0.0 <= float(self.freshness) <= 1.0:
            raise ValueError("freshness must be in [0, 1]")
        if self.touch_count < 0:
            raise ValueError("touch_count cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class TouchEvent(DomainEvent):
    zone_id: str
    touch_at: datetime
    observed_at: datetime
    depth: TouchDepth
    sweep_price: float

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        _require_text(self.zone_id, "zone_id")
        _require_utc(self.touch_at, "touch_at")
        _require_utc(self.observed_at, "observed_at")
        if self.occurred_at != self.touch_at or self.known_at != self.observed_at:
            raise ValueError("touch occurred_at/known_at must equal touch_at/observed_at")
        _require_price(self.sweep_price, "sweep_price")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReclaimEvent(DomainEvent):
    zone_id: str
    touch_event_id: str
    side: Side
    reclaim_price: float

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        _require_text(self.zone_id, "zone_id")
        _require_text(self.touch_event_id, "touch_event_id")
        _require_price(self.reclaim_price, "reclaim_price")


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalEvent(DomainEvent):
    strategy_id: str
    side: Side
    zone_id: str
    reclaim_event_id: str
    signal_at: datetime
    entry_reference: float
    stop_loss: float
    take_profit: float
    score: float

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        _require_text(self.strategy_id, "strategy_id")
        _require_text(self.zone_id, "zone_id")
        _require_text(self.reclaim_event_id, "reclaim_event_id")
        _require_utc(self.signal_at, "signal_at")
        if self.signal_at != self.known_at:
            raise ValueError("signal_at must equal known_at")
        for name in ("entry_reference", "stop_loss", "take_profit"):
            _require_price(getattr(self, name), name)
        if self.side is Side.BUY and not self.stop_loss < self.entry_reference < self.take_profit:
            raise ValueError("buy geometry must satisfy stop < entry < target")
        if self.side is Side.SELL and not self.take_profit < self.entry_reference < self.stop_loss:
            raise ValueError("sell geometry must satisfy target < entry < stop")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskDecisionEvent(DomainEvent):
    signal_id: str
    approved: bool
    reason: str
    risk_amount: float
    approved_volume: float

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        _require_text(self.signal_id, "signal_id")
        _require_text(self.reason, "reason")
        for name in ("risk_amount", "approved_volume"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.approved and (self.risk_amount <= 0 or self.approved_volume <= 0):
            raise ValueError("approved risk decisions require positive risk and volume")
        if not self.approved and self.approved_volume != 0:
            raise ValueError("rejected risk decisions must approve zero volume")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderIntentEvent(DomainEvent):
    signal_id: str
    risk_decision_id: str
    idempotency_key: str
    side: Side
    volume: float
    stop_loss: float
    take_profit: float

    def __post_init__(self) -> None:
        DomainEvent.__post_init__(self)
        for name in ("signal_id", "risk_decision_id", "idempotency_key"):
            _require_text(getattr(self, name), name)
        if not isfinite(float(self.volume)) or self.volume <= 0:
            raise ValueError("volume must be finite and positive")
        _require_price(self.stop_loss, "stop_loss")
        _require_price(self.take_profit, "take_profit")
