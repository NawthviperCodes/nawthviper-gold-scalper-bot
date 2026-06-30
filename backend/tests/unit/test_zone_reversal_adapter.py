import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from strategies import ZoneReversalConfig, ZoneReversalStrategy
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


UTC = timezone.utc
AS_OF = datetime(2026, 1, 10, 12, tzinfo=UTC)


def series(timeframe, count, minutes):
    bars = []
    start = AS_OF - timedelta(minutes=count * minutes)
    for idx in range(count):
        opened = start + timedelta(minutes=idx * minutes)
        closed = opened + timedelta(minutes=minutes)
        price = 100.0 + idx * 0.1
        bars.append(
            MarketBar(
                symbol="XAUUSDz",
                timeframe=timeframe,
                opened_at=opened,
                closed_at=closed,
                known_at=closed,
                open=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price + 0.1,
                tick_volume=100,
            )
        )
    return TimeframeSeries(timeframe=timeframe, bars=tuple(bars))


def demand_zone():
    created = AS_OF - timedelta(days=3)
    activated = AS_OF - timedelta(days=2)
    return ZoneEvent(
        event_id=stable_event_id("zone", "XAUUSDz", created, activated, 105.0, 107.0),
        symbol="XAUUSDz",
        occurred_at=created,
        known_at=activated,
        zone_type=ZoneType.DEMAND,
        created_at=created,
        detected_at=activated,
        activated_at=activated,
        bottom=105.0,
        top=107.0,
        quality="A",
        freshness=1.0,
        touch_count=0,
    )


def snapshot(*, with_quote=True, with_position=False):
    quote = (
        MarketQuote(
            symbol="XAUUSDz",
            bid=120.0,
            ask=120.1,
            occurred_at=AS_OF,
            known_at=AS_OF,
        )
        if with_quote
        else None
    )
    positions = ()
    if with_position:
        positions = (
            PositionState(
                position_id="position-1",
                symbol="XAUUSDz",
                side=Side.BUY,
                volume=0.1,
                entry_price=118.0,
                opened_at=AS_OF - timedelta(hours=1),
                known_at=AS_OF,
            ),
        )
    return StrategySnapshot(
        symbol="XAUUSDz",
        as_of=AS_OF,
        series=(series("H4", 220, 240), series("H1", 100, 60), series("M5", 80, 5)),
        active_zones=(demand_zone(),),
        quote=quote,
        positions=positions,
    )


class FakeLegacyDecision:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (
            [
                {
                    "side": "buy",
                    "entry": 120.0,
                    "sl": 115.0,
                    "tp": 130.0,
                    "zone": {"mid": 105.0},
                    "strategy": kwargs["strategy_mode"],
                    "reason": "buy_reclaim",
                    "confidence": 0.7,
                }
            ],
            [],
        )


class ZoneReversalAdapterTests(unittest.TestCase):
    def test_adapter_maps_snapshot_to_legacy_inputs_and_back(self):
        legacy = FakeLegacyDecision()
        config = ZoneReversalConfig.from_thresholds(
            point=0.01,
            tp_ratio=2.0,
            thresholds={"MAX_TOUCH_ALLOWED": 2, "MIN_RR_FILTER": 1.3},
        )
        strategy = ZoneReversalStrategy(config, decision_fn=legacy)

        signals = StrategyRunner().run(strategy, snapshot())

        self.assertEqual(1, len(signals))
        signal = signals[0]
        self.assertEqual(demand_zone().event_id, signal.zone_id)
        self.assertEqual(Side.BUY, signal.side)
        self.assertEqual(AS_OF, signal.signal_at)
        self.assertEqual(120.0, signal.entry_reference)
        self.assertEqual(115.0, signal.stop_loss)
        self.assertEqual(130.0, signal.take_profit)
        self.assertEqual(120.0, legacy.calls[0]["current_price"])
        self.assertAlmostEqual(0.1, legacy.calls[0]["spread"])
        self.assertEqual(
            105.0,
            legacy.calls[0]["demand_zones"][0]["displacement_origin_low"],
        )
        self.assertEqual(
            107.0,
            legacy.calls[0]["demand_zones"][0]["displacement_origin_high"],
        )
        self.assertEqual(5, len(legacy.calls[0]["m5_candles_for_patterns"]))
        self.assertTrue(legacy.calls[0]["strategy_active_override"])

    def test_same_snapshot_produces_same_canonical_signal_id(self):
        config = ZoneReversalConfig.from_thresholds(point=0.01, thresholds={})
        strategy = ZoneReversalStrategy(config, decision_fn=FakeLegacyDecision())
        runner = StrategyRunner()
        snap = snapshot()

        first = runner.run(strategy, snap)
        second = runner.run(strategy, snap)

        self.assertEqual(first, second)

    def test_position_state_is_mapped_to_legacy_conflict_input(self):
        legacy = FakeLegacyDecision()
        strategy = ZoneReversalStrategy(
            ZoneReversalConfig.from_thresholds(point=0.01, thresholds={}),
            decision_fn=legacy,
        )

        strategy.evaluate(snapshot(with_position=True))

        self.assertEqual("buy", legacy.calls[0]["active_trades"]["XAUUSDz"]["side"])
        self.assertEqual("position-1", legacy.calls[0]["active_trades"]["XAUUSDz"]["ticket"])

    def test_missing_quote_fails_closed(self):
        strategy = ZoneReversalStrategy(
            ZoneReversalConfig.from_thresholds(point=0.01, thresholds={}),
            decision_fn=FakeLegacyDecision(),
        )
        with self.assertRaisesRegex(ValueError, "requires a causal market quote"):
            strategy.evaluate(snapshot(with_quote=False))

    def test_strategy_activity_override_is_explicit(self):
        legacy = FakeLegacyDecision()
        strategy = ZoneReversalStrategy(
            ZoneReversalConfig.from_thresholds(
                point=0.01,
                thresholds={},
                strategy_active=False,
            ),
            decision_fn=legacy,
        )

        strategy.evaluate(snapshot())

        self.assertFalse(legacy.calls[0]["strategy_active_override"])


if __name__ == "__main__":
    unittest.main()
