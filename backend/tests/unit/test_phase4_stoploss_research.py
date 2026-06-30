import sys
import unittest
from pathlib import Path

import pandas as pd


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from phase4_stoploss_research import evaluate_stop_model, run_symbol_study


class Phase4CausalClockTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "reaction_horizon_bars": 2,
            "tp_target_mode": "fixed_r_multiple",
            "fixed_r_multiple": 2.0,
            "wick_buffer_mults": [0.15, 0.25],
            "deep_buffer_mult": 0.25,
        }

    def test_pre_entry_touch_high_cannot_hit_post_entry_target(self):
        prices = pd.DataFrame(
            [
                # This high would create a fictional target hit if the study
                # evaluated the completed touch candle after entering at close.
                {"time": "2026-01-01 10:00", "open": 95.0, "high": 140.0, "low": 90.0, "close": 100.0},
                {"time": "2026-01-01 11:00", "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0},
                {"time": "2026-01-01 12:00", "open": 102.0, "high": 106.0, "low": 100.0, "close": 103.0},
            ]
        )
        row = pd.Series({"zone_type": "demand", "width": 5.0, "atr_at_formation": 10.0})

        result = evaluate_stop_model(
            prices,
            entry_idx=1,
            row=row,
            entry=100.0,
            stop=88.5,
            horizon=2,
            cfg=self.cfg,
        )

        self.assertEqual("no_resolution", result["status"])
        self.assertFalse(result["target_hit"])

    def test_study_enters_on_bar_after_touch_and_emits_event_timestamps(self):
        prices = pd.DataFrame(
            [
                {"time": pd.Timestamp("2026-01-01 10:00"), "open": 95.0, "high": 110.0, "low": 90.0, "close": 100.0},
                {"time": pd.Timestamp("2026-01-01 11:00"), "open": 101.0, "high": 104.0, "low": 99.0, "close": 102.0},
                {"time": pd.Timestamp("2026-01-01 12:00"), "open": 102.0, "high": 105.0, "low": 100.0, "close": 103.0},
            ]
        )
        zones = pd.DataFrame(
            [{
                "touched": True,
                "first_touch_idx": 0,
                "zone_type": "demand",
                "zone_bottom": 90.0,
                "zone_top": 95.0,
                "width": 5.0,
                "atr_at_formation": 10.0,
            }]
        )

        events, _ = run_symbol_study("TEST", prices, zones, self.cfg)

        self.assertFalse(events.empty)
        event = events.iloc[0]
        self.assertEqual(0, event["touch_idx"])
        self.assertEqual(1, event["signal_idx"])
        self.assertEqual(1, event["entry_idx"])
        self.assertEqual(101.0, event["entry_price"])
        self.assertEqual(pd.Timestamp("2026-01-01 10:00"), event["touch_timestamp"])
        self.assertEqual(pd.Timestamp("2026-01-01 11:00"), event["signal_timestamp"])
        self.assertEqual(pd.Timestamp("2026-01-01 11:00"), event["entry_timestamp"])

    def test_same_entry_bar_collision_remains_conservative(self):
        prices = pd.DataFrame(
            [{"time": "2026-01-01 11:00", "open": 100.0, "high": 125.0, "low": 85.0, "close": 100.0}]
        )
        row = pd.Series({"zone_type": "demand", "width": 5.0, "atr_at_formation": 10.0})

        result = evaluate_stop_model(
            prices,
            entry_idx=0,
            row=row,
            entry=100.0,
            stop=90.0,
            horizon=1,
            cfg=self.cfg,
        )

        self.assertEqual("ambiguous_same_bar", result["status"])
        self.assertEqual(-1.0, result["r_multiple"])


if __name__ == "__main__":
    unittest.main()
