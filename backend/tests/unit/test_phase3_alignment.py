import sys
import unittest
from pathlib import Path

import pandas as pd


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from phase3_candlestick_research import (
    enrich_symbol,
    find_touch_m5_index,
    summarize_symbol,
)


def m5_frame():
    times = pd.date_range("2026-01-01 10:00", periods=14, freq="5min")
    return pd.DataFrame(
        {
            "time": times,
            "open": [100.0] * len(times),
            "high": [102.0] * len(times),
            "low": [99.0] * len(times),
            "close": [101.0] * len(times),
        }
    )


def zone_row(**overrides):
    row = {
        "touched": True,
        "reaction_outcome": "immediate_reversal",
        "first_touch_time": pd.Timestamp("2026-01-01 10:00"),
        "touch_observed_at": pd.Timestamp("2026-01-01 11:00"),
        "zone_bottom": 90.0,
        "zone_top": 95.0,
        "zone_type": "demand",
        "touches": 0,
    }
    row.update(overrides)
    return row


class Phase3AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"touch_window_minutes": 60, "lookback_candles": 3}

    def test_last_m5_candle_closing_at_h1_observation_is_allowed(self):
        m5 = m5_frame()
        m5.loc[11, ["high", "low"]] = [101.0, 94.0]  # 10:55 -> closes 11:00

        idx = find_touch_m5_index(
            m5,
            pd.Timestamp("2026-01-01 10:00"),
            90.0,
            95.0,
            touch_observed_at=pd.Timestamp("2026-01-01 11:00"),
        )

        self.assertEqual(11, idx)

    def test_m5_candle_closing_after_observation_is_rejected(self):
        m5 = m5_frame()
        m5.loc[12, ["high", "low"]] = [101.0, 94.0]  # 11:00 -> closes 11:05

        idx = find_touch_m5_index(
            m5,
            pd.Timestamp("2026-01-01 10:00"),
            90.0,
            95.0,
            touch_observed_at=pd.Timestamp("2026-01-01 11:00"),
        )

        self.assertIsNone(idx)

    def test_no_zone_overlap_has_no_nearest_candle_fallback(self):
        m5 = m5_frame()
        idx = find_touch_m5_index(
            m5,
            pd.Timestamp("2026-01-01 10:00"),
            90.0,
            95.0,
            touch_observed_at=pd.Timestamp("2026-01-01 11:00"),
        )
        self.assertIsNone(idx)

    def test_enrichment_records_when_pattern_became_knowable(self):
        m5 = m5_frame()
        m5.loc[11, ["open", "high", "low", "close"]] = [96.0, 101.0, 94.0, 100.0]
        zones = pd.DataFrame([zone_row()])

        enriched, summary = enrich_symbol("TEST", zones, m5, self.cfg)
        event = enriched.iloc[0]

        self.assertEqual("m5_touch_aligned", event["m5_alignment_status"])
        self.assertEqual(11, event["m5_touch_idx"])
        self.assertEqual(pd.Timestamp("2026-01-01 11:00"), event["pattern_knowledge_time"])
        self.assertEqual("explicit_touch_observed_at", event["touch_boundary_source"])
        self.assertEqual(1, summary["pattern_evaluable_touches"])

    def test_right_censored_event_is_not_pattern_evaluable(self):
        m5 = m5_frame()
        m5.loc[11, ["high", "low"]] = [101.0, 94.0]
        zones = pd.DataFrame([zone_row(reaction_outcome="right_censored")])

        enriched, summary = enrich_symbol("TEST", zones, m5, self.cfg)

        self.assertEqual("excluded_right_censored", enriched.iloc[0]["m5_alignment_status"])
        self.assertEqual(1, summary["right_censored_count"])
        self.assertEqual(0, summary["evaluable_touched_zones"])
        self.assertEqual(0, summary["pattern_evaluable_touches"])

    def test_summary_separates_censoring_from_alignment_failure(self):
        enriched = pd.DataFrame(
            [
                {
                    "touched": True,
                    "reaction_outcome": "immediate_reversal",
                    "m5_alignment_status": "m5_touch_aligned",
                    "pattern_name": "no_pattern",
                },
                {
                    "touched": True,
                    "reaction_outcome": "breakout",
                    "m5_alignment_status": "no_m5_zone_overlap",
                    "pattern_name": "no_pattern",
                },
                {
                    "touched": True,
                    "reaction_outcome": "right_censored",
                    "m5_alignment_status": "excluded_right_censored",
                    "pattern_name": "no_pattern",
                },
            ]
        )

        summary = summarize_symbol(enriched)

        self.assertEqual(3, summary["touched_zones"])
        self.assertEqual(2, summary["evaluable_touched_zones"])
        self.assertEqual(1, summary["right_censored_count"])
        self.assertEqual(1, summary["pattern_evaluable_touches"])
        self.assertEqual(1, summary["m5_alignment_failure_count"])


if __name__ == "__main__":
    unittest.main()
