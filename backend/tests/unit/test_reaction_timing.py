import sys
import types
import unittest
from pathlib import Path

import pandas as pd


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

# zone_event_study imports the detector, but these timing tests do not require
# JIT compilation or the third-party ATR implementation.
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        numba_stub = types.ModuleType("numba")
        numba_stub.jit = lambda *args, **kwargs: (args[0] if args and callable(args[0]) else lambda func: func)
        sys.modules["numba"] = numba_stub

try:
    from ta.volatility import AverageTrueRange  # noqa: F401
except ModuleNotFoundError:
    ta_stub = types.ModuleType("ta")
    volatility_stub = types.ModuleType("ta.volatility")

    class AverageTrueRange:
        def __init__(self, high, low, close, window=14):
            self._atr = (high - low).rolling(window, min_periods=1).mean()

        def average_true_range(self):
            return self._atr

    volatility_stub.AverageTrueRange = AverageTrueRange
    ta_stub.volatility = volatility_stub
    sys.modules["ta"] = ta_stub
    sys.modules["ta.volatility"] = volatility_stub

from zone_event_study import bar_close_timestamp, classify_reaction
from zone_reaction_research import (
    build_symbol_summary,
    derive_reaction_family,
    grouped_reaction_rates,
)


DEMAND_ZONE = {
    "type": "demand",
    "bottom": 90.0,
    "top": 95.0,
    "width": 5.0,
    "atr_at_formation": 0.0,
}


def prices(rows):
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=len(frame), freq="h"))
    return frame


class ReactionClockTests(unittest.TestCase):
    def test_touch_bar_high_cannot_be_post_touch_reversal(self):
        df = prices(
            [
                [98.0, 120.0, 92.0, 100.0],  # touch; target was reached before observation
                [100.0, 104.0, 98.0, 102.0],
                [102.0, 104.0, 99.0, 103.0],
            ]
        )
        outcome, idx = classify_reaction(df, 0, DEMAND_ZONE, horizon=2)
        self.assertEqual("no_clear_reaction", outcome)
        self.assertIsNone(idx)

    def test_first_post_touch_bar_is_immediate_reversal(self):
        df = prices(
            [
                [98.0, 103.0, 92.0, 100.0],
                [100.0, 106.0, 98.0, 105.0],
                [105.0, 107.0, 103.0, 106.0],
            ]
        )
        outcome, idx = classify_reaction(df, 0, DEMAND_ZONE, horizon=2)
        self.assertEqual("immediate_reversal", outcome)
        self.assertEqual(1, idx)

    def test_later_post_touch_bar_is_consolidation_reversal(self):
        df = prices(
            [
                [98.0, 103.0, 92.0, 100.0],
                [100.0, 104.0, 98.0, 102.0],
                [102.0, 106.0, 99.0, 105.0],
            ]
        )
        outcome, idx = classify_reaction(df, 0, DEMAND_ZONE, horizon=2)
        self.assertEqual("consolidation_then_reversal", outcome)
        self.assertEqual(2, idx)

    def test_touch_bar_close_through_is_confirmed_breakout(self):
        df = prices(
            [
                [95.0, 100.0, 85.0, 89.0],
                [89.0, 92.0, 87.0, 90.0],
            ]
        )
        outcome, idx = classify_reaction(df, 0, DEMAND_ZONE, horizon=2)
        self.assertEqual("breakout", outcome)
        self.assertEqual(0, idx)

    def test_incomplete_forward_window_is_right_censored(self):
        df = prices(
            [
                [100.0, 102.0, 99.0, 101.0],
                [98.0, 103.0, 92.0, 100.0],
                [100.0, 104.0, 98.0, 102.0],
            ]
        )
        outcome, idx = classify_reaction(df, 1, DEMAND_ZONE, horizon=3)
        self.assertEqual("right_censored", outcome)
        self.assertIsNone(idx)

    def test_bar_close_timestamp_uses_next_bar_open(self):
        df = prices(
            [
                [100.0, 102.0, 99.0, 101.0],
                [101.0, 103.0, 100.0, 102.0],
            ]
        )
        self.assertEqual(df.iloc[1]["time"], bar_close_timestamp(df, 0))

    def test_censored_rows_are_excluded_from_performance_denominator(self):
        df = pd.DataFrame(
            [
                {"touched": True, "reaction_outcome": "immediate_reversal", "zone_type": "demand"},
                {"touched": True, "reaction_outcome": "breakout", "zone_type": "demand"},
                {"touched": True, "reaction_outcome": "right_censored", "zone_type": "demand"},
            ]
        )
        df["reaction_family"] = df["reaction_outcome"].map(derive_reaction_family)

        summary = build_symbol_summary("TEST", df)
        grouped = grouped_reaction_rates(df, "zone_type").iloc[0]

        self.assertEqual(3, summary["touched_zones"])
        self.assertEqual(2, summary["evaluable_touched_zones"])
        self.assertEqual(1, summary["right_censored_count"])
        self.assertEqual(50.0, summary["reversal_rate_evaluable_pct"])
        self.assertEqual(2, grouped["evaluable_count"])
        self.assertEqual(50.0, grouped["reversal_rate_pct"])


if __name__ == "__main__":
    unittest.main()
