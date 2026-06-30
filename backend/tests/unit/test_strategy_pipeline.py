import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from strategies import (
    ZoneReversalConfig,
    build_zone_reversal_snapshot,
    closed_bars_frame,
    closed_bars_slice,
    normalize_pipeline_mode,
    quote_rejection_reason,
    run_zone_reversal_pipeline,
    wilder_atr,
)
from strategies.pipeline import dataframe_to_series


UTC = timezone.utc


def frame(timeframe_minutes, count, *, end_at, price=100.0, step=0.0):
    first = end_at - timedelta(minutes=timeframe_minutes * count)
    return pd.DataFrame([
        {
            "time": first + timedelta(minutes=timeframe_minutes * i),
            "open": price + step * i,
            "high": price + step * i + 1.0,
            "low": price + step * i - 1.0,
            "close": price + step * i + 0.25,
            "tick_volume": 10,
        }
        for i in range(count)
    ])


def demand_zone(as_of):
    return {
        "created_at": as_of - timedelta(hours=20),
        "detected_at": as_of - timedelta(hours=17),
        "activated_at": as_of - timedelta(hours=17),
        "bottom": 99.0,
        "top": 100.0,
        "price": 99.0,
        "quality_bucket": "A",
        "freshness_score": 1.0,
        "touches": 0,
    }


def fixed_signal():
    return {
        "side": "buy",
        "entry": 100.0,
        "sl": 98.0,
        "tp": 104.0,
        "zone": {"mid": 99.0},
        "strategy": "standard",
        "reason": "buy_reclaim",
        "confidence": 0.75,
    }


class StrategyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.as_of = datetime(2026, 1, 10, 12, tzinfo=UTC)

    def snapshot(self):
        return build_zone_reversal_snapshot(
            symbol="TEST",
            as_of=self.as_of,
            bid=100.0,
            ask=100.1,
            h4=frame(240, 200, end_at=self.as_of, step=0.01),
            h1=frame(60, 80, end_at=self.as_of),
            m5=frame(5, 60, end_at=self.as_of),
            demand_zones=[demand_zone(self.as_of)],
            supply_zones=[],
            active_trades={},
        )

    def config(self):
        return ZoneReversalConfig.from_thresholds(
            strategy_id="standard",
            point=0.01,
            pattern_bars=15,
            thresholds={},
        )

    def test_default_project_config_matches_locked_demo_deployment(self):
        config_path = Path(__file__).parents[2] / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual("legacy", raw["BotSettings"]["STRATEGY_PIPELINE_MODE"])
        self.assertFalse(raw["BotSettings"]["DRY_RUN"])
        self.assertEqual(
            ["XAUUSDz", "EURUSDz"], raw["BotSettings"]["SYMBOLS"]
        )
        self.assertEqual(120, raw["BotSettings"]["MAX_QUOTE_AGE_SECONDS"])

    def test_pipeline_mode_rejects_typo_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "STRATEGY_PIPELINE_MODE"):
            normalize_pipeline_mode("canoncial")

    def test_dataframe_conversion_excludes_still_open_bar(self):
        bars = frame(5, 2, end_at=self.as_of + timedelta(minutes=5))
        series = dataframe_to_series(
            symbol="TEST", timeframe="M5", frame=bars, as_of=self.as_of
        )
        self.assertEqual(1, len(series.bars))
        self.assertLessEqual(series.bars[-1].closed_at, self.as_of)

    def test_closed_m5_window_uses_close_time_and_includes_full_h1_interval(self):
        bars = frame(5, 13, end_at=self.as_of + timedelta(minutes=5))
        selected = closed_bars_frame(
            bars, timeframe="M5", as_of=self.as_of, tail=80
        )
        self.assertEqual(12, len(selected))
        self.assertEqual(self.as_of - timedelta(minutes=5), selected.iloc[-1]["time"])

    def test_closed_slice_is_boundary_exact_and_rejects_unsorted_input(self):
        opened = pd.DatetimeIndex(pd.to_datetime([
            self.as_of - timedelta(minutes=10),
            self.as_of - timedelta(minutes=5),
            self.as_of,
        ], utc=True))
        selection = closed_bars_slice(
            opened, timeframe="M5", as_of=self.as_of, tail=80
        )
        self.assertEqual(slice(0, 2), selection)
        with self.assertRaisesRegex(ValueError, "sorted"):
            closed_bars_slice(
                opened[::-1], timeframe="M5", as_of=self.as_of, tail=80
            )

    def test_h4_bar_is_unavailable_until_four_hours_after_open(self):
        bars = pd.DataFrame([{
            "time": self.as_of - timedelta(hours=3),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 1,
        }])
        self.assertTrue(closed_bars_frame(bars, timeframe="H4", as_of=self.as_of).empty)
        available = closed_bars_frame(
            bars, timeframe="H4", as_of=self.as_of + timedelta(hours=1)
        )
        self.assertEqual(1, len(available))

    def test_replay_wiring_uses_shared_m5_atr_and_current_closed_h1(self):
        source = (Path(__file__).parents[2] / "backtester.py").read_text(encoding="utf-8")
        self.assertIn("atr_val   = wilder_atr(m5_window)", source)
        self.assertIn("h1_atr = wilder_atr(h1_window)", source)
        self.assertIn("last_closed_h1 = h1_window.iloc[-1]", source)
        self.assertNotIn("m5['time'] <= candle['time']", source)
        self.assertNotIn("h4['time'] <= candle['time']", source)
        self.assertIn("Causal M5 replay matching the live strategy's decision cadence", source)
        self.assertIn("m5[(m5['time'] >= range_start)", source)
        self.assertIn("timeframe_close_delta(\n            cfg.get('tf_confirm', 'M5')", source)
        self.assertIn("strategy_active_override=True", source)
        self.assertIn("emit_telemetry=False", source)

    def test_live_wiring_filters_forming_bars_and_uses_shared_atr(self):
        source = (Path(__file__).parents[2] / "scalper_strategy_engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("h4_df = get_closed_data(symbol, TIMEFRAME_HTF, 250, as_of)", source)
        self.assertIn(
            "h1_df = get_closed_data(symbol, TIMEFRAME_ZONE, ZONE_LOOKBACK, as_of)", source
        )
        self.assertIn("m5_df = get_closed_data(symbol, TIMEFRAME_CONFIRM, 80, as_of)", source)
        self.assertIn("h1_atr = wilder_atr(h1_df)", source)
        self.assertIn("atr = wilder_atr(m5_df)", source)
        self.assertIn("last_closed_h1=h1_df.iloc[-1]", source)
        self.assertNotIn("AverageTrueRange", source)
        self.assertIn("if not DRY_RUN:\n            trail_sl(symbol, MAGIC)", source)
        self.assertIn("not DRY_RUN\n                and existing['side'] != side", source)
        self.assertIn("quote_reason = quote_rejection_reason(", source)

    def test_quote_validation_rejects_bad_geometry_and_missing_timestamp(self):
        self.assertEqual(
            "invalid",
            quote_rejection_reason(
                bid=0,
                ask=0,
                occurred_at=self.as_of,
                observed_at=self.as_of,
                max_age_seconds=120,
            ),
        )
        self.assertEqual(
            "invalid",
            quote_rejection_reason(
                bid=101,
                ask=100,
                occurred_at=self.as_of,
                observed_at=self.as_of,
                max_age_seconds=120,
            ),
        )
        self.assertEqual(
            "invalid",
            quote_rejection_reason(
                bid=100,
                ask=101,
                occurred_at=None,
                observed_at=self.as_of,
                max_age_seconds=120,
            ),
        )

    def test_quote_validation_rejects_stale_and_future_quotes(self):
        self.assertEqual(
            "stale",
            quote_rejection_reason(
                bid=100,
                ask=101,
                occurred_at=self.as_of - timedelta(seconds=121),
                observed_at=self.as_of,
                max_age_seconds=120,
            ),
        )
        self.assertEqual(
            "future",
            quote_rejection_reason(
                bid=100,
                ask=101,
                occurred_at=self.as_of + timedelta(seconds=6),
                observed_at=self.as_of,
                max_age_seconds=120,
            ),
        )
        self.assertIsNone(
            quote_rejection_reason(
                bid=100,
                ask=101,
                occurred_at=self.as_of - timedelta(seconds=120),
                observed_at=self.as_of,
                max_age_seconds=120,
            )
        )

    def test_shared_wilder_atr_is_positive_and_deterministic(self):
        bars = frame(5, 80, end_at=self.as_of, step=0.02)
        first = wilder_atr(bars)
        second = wilder_atr(bars.copy())
        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)

    def test_legacy_mode_does_not_require_or_build_canonical_state(self):
        calls = []

        def legacy(**kwargs):
            calls.append(kwargs)
            return [fixed_signal()], ["legacy-rejection"]

        result = run_zone_reversal_pipeline(
            mode="legacy", legacy_kwargs={"sentinel": 7}, decision_fn=legacy
        )
        self.assertEqual(([fixed_signal()], ["legacy-rejection"]), result)
        self.assertEqual([{"sentinel": 7}], calls)

    def test_shadow_failure_never_changes_legacy_result(self):
        reports = []

        def decision(**kwargs):
            if "sentinel" in kwargs:
                return [fixed_signal()], []
            raise RuntimeError("canonical unavailable")

        result = run_zone_reversal_pipeline(
            mode="shadow",
            legacy_kwargs={"sentinel": 7},
            snapshot=self.snapshot(),
            config=self.config(),
            decision_fn=decision,
            parity_sink=reports.append,
        )
        self.assertEqual([fixed_signal()], result[0])
        self.assertFalse(reports[0].matched)
        self.assertIn("canonical unavailable", reports[0].canonical_error)

    def test_shadow_reports_matching_decisions(self):
        reports = []
        calls = []

        def decision(**kwargs):
            calls.append(kwargs)
            return [fixed_signal()], []

        result = run_zone_reversal_pipeline(
            mode="shadow",
            legacy_kwargs={"symbol": "TEST"},
            snapshot=self.snapshot(),
            config=self.config(),
            decision_fn=decision,
            parity_sink=reports.append,
        )
        self.assertEqual([fixed_signal()], result[0])
        self.assertTrue(reports[0].matched)
        self.assertFalse(calls[1]["emit_telemetry"])

    def test_canonical_mode_preserves_legacy_output_contract(self):
        def decision(**kwargs):
            return [fixed_signal()], []

        signals, rejections = run_zone_reversal_pipeline(
            mode="canonical",
            legacy_kwargs={},
            snapshot=self.snapshot(),
            config=self.config(),
            decision_fn=decision,
        )
        self.assertEqual([fixed_signal()], signals)
        self.assertEqual([], rejections)


if __name__ == "__main__":
    unittest.main()
