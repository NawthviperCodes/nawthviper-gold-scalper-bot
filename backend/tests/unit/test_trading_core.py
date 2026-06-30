import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from trading_core import (
    MarketBar,
    Side,
    SignalEvent,
    StrategyRunner,
    StrategySnapshot,
    TimeframeSeries,
    ZoneEvent,
    ZoneType,
    stable_event_id,
)


UTC = timezone.utc


def at(hour, minute=0):
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def bar(opened=10, closed=11, known=11):
    return MarketBar(
        symbol="XAUUSDz",
        timeframe="H1",
        opened_at=at(opened),
        closed_at=at(closed),
        known_at=at(known),
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        tick_volume=10,
    )


def zone():
    created = at(6)
    activated = at(11)
    return ZoneEvent(
        event_id=stable_event_id("zone", "XAUUSDz", created, activated, 90.0, 95.0),
        symbol="XAUUSDz",
        occurred_at=created,
        known_at=activated,
        zone_type=ZoneType.DEMAND,
        created_at=created,
        detected_at=activated,
        activated_at=activated,
        bottom=90.0,
        top=95.0,
        quality="A",
        freshness=1.0,
    )


class DeterministicStrategy:
    strategy_id = "zone_reversal_v2"

    def evaluate(self, snapshot):
        z = snapshot.active_zones[0]
        event_id = stable_event_id("signal", snapshot.symbol, self.strategy_id, z.event_id, snapshot.as_of)
        return (
            SignalEvent(
                event_id=event_id,
                symbol=snapshot.symbol,
                occurred_at=snapshot.as_of,
                known_at=snapshot.as_of,
                parent_event_id=z.event_id,
                strategy_id=self.strategy_id,
                side=Side.BUY,
                zone_id=z.event_id,
                reclaim_event_id="reclaim-1",
                signal_at=snapshot.as_of,
                entry_reference=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=0.7,
            ),
        )


class TradingCoreTests(unittest.TestCase):
    def test_event_ids_are_deterministic(self):
        first = stable_event_id("signal", "XAUUSDz", at(11), "zone-1")
        second = stable_event_id("signal", "XAUUSDz", at(11), "zone-1")
        changed = stable_event_id("signal", "XAUUSDz", at(12), "zone-1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_domain_objects_are_immutable(self):
        item = bar()
        with self.assertRaises(FrozenInstanceError):
            item.close = 999.0

    def test_bar_cannot_be_known_before_close(self):
        with self.assertRaisesRegex(ValueError, "before it closes"):
            bar(opened=10, closed=11, known=10)

    def test_naive_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            MarketBar(
                symbol="XAUUSDz",
                timeframe="H1",
                opened_at=datetime(2026, 1, 1, 10),
                closed_at=datetime(2026, 1, 1, 11),
                known_at=datetime(2026, 1, 1, 11),
                open=100,
                high=101,
                low=99,
                close=100,
            )

    def test_zone_activation_contract_is_explicit(self):
        item = zone()
        self.assertEqual(item.created_at, item.occurred_at)
        self.assertEqual(item.activated_at, item.known_at)
        self.assertGreater(item.activated_at, item.created_at)

    def test_snapshot_rejects_future_bar(self):
        future_bar = MarketBar(
            symbol="XAUUSDz",
            timeframe="H1",
            opened_at=at(11),
            closed_at=at(12),
            known_at=at(12),
            open=102,
            high=106,
            low=101,
            close=105,
        )
        with self.assertRaisesRegex(ValueError, "unknown at as_of"):
            StrategySnapshot(
                symbol="XAUUSDz",
                as_of=at(11),
                series=(TimeframeSeries(timeframe="H1", bars=(bar(), future_bar)),),
                active_zones=(zone(),),
            )

    def test_runner_produces_identical_live_and_replay_decisions(self):
        snapshot = StrategySnapshot(
            symbol="XAUUSDz",
            as_of=at(11),
            series=(TimeframeSeries(timeframe="H1", bars=(bar(),)),),
            active_zones=(zone(),),
        )
        strategy = DeterministicStrategy()
        runner = StrategyRunner()

        live_result = runner.run(strategy, snapshot)
        replay_result = runner.run(strategy, snapshot)

        self.assertEqual(live_result, replay_result)
        self.assertEqual(snapshot.as_of, live_result[0].known_at)

    def test_runner_rejects_signal_created_on_wrong_clock(self):
        snapshot = StrategySnapshot(
            symbol="XAUUSDz",
            as_of=at(11),
            series=(TimeframeSeries(timeframe="H1", bars=(bar(),)),),
            active_zones=(zone(),),
        )

        class LateStrategy(DeterministicStrategy):
            def evaluate(self, snapshot):
                signal = super().evaluate(snapshot)[0]
                late = snapshot.as_of + timedelta(minutes=1)
                return (
                    SignalEvent(
                        event_id="late-signal",
                        symbol=signal.symbol,
                        occurred_at=late,
                        known_at=late,
                        strategy_id=self.strategy_id,
                        side=signal.side,
                        zone_id=signal.zone_id,
                        reclaim_event_id=signal.reclaim_event_id,
                        signal_at=late,
                        entry_reference=signal.entry_reference,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        score=signal.score,
                    ),
                )

        with self.assertRaisesRegex(ValueError, "snapshot.as_of"):
            StrategyRunner().run(LateStrategy(), snapshot)


if __name__ == "__main__":
    unittest.main()
