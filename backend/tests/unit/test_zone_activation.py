import sys
import types
import unittest
from pathlib import Path

import pandas as pd


BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

# The repository currently imports numba but does not declare it in
# requirements.txt.  Causal unit tests do not need machine-code compilation,
# so use an identity decorator when the test runtime lacks that optional JIT.
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    numba_stub = types.ModuleType("numba")

    def identity_jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func

    numba_stub.jit = identity_jit
    sys.modules["numba"] = numba_stub

try:
    from ta.volatility import AverageTrueRange  # noqa: F401
except ModuleNotFoundError:
    ta_stub = types.ModuleType("ta")
    volatility_stub = types.ModuleType("ta.volatility")

    class AverageTrueRange:
        def __init__(self, high, low, close, window=14):
            previous_close = close.shift(1)
            true_range = pd.concat(
                [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
                axis=1,
            ).max(axis=1)
            self._atr = true_range.rolling(window, min_periods=1).mean()

        def average_true_range(self):
            return self._atr

    volatility_stub.AverageTrueRange = AverageTrueRange
    ta_stub.volatility = volatility_stub
    sys.modules["ta"] = ta_stub
    sys.modules["ta.volatility"] = volatility_stub

from zone_detector import detect_zones
from zone_event_study import run_symbol_study


def market_frame(length=100):
    times = pd.date_range("2026-01-01", periods=length, freq="h")
    return pd.DataFrame(
        {
            "time": times,
            "open": [100.8] * length,
            "high": [102.0] * length,
            "low": [100.0] * length,
            "close": [101.0] * length,
        }
    )


def add_demand_candidate(df, pivot_idx, low=95.0):
    df.loc[pivot_idx, ["open", "high", "low", "close"]] = [low + 0.5, low + 1.0, low, low + 0.8]
    # Four completed departure bars are needed.  With zone_size=2 and
    # departure_bars=4, the zone becomes knowable at pivot_idx + 4.
    for idx in range(pivot_idx + 1, pivot_idx + 5):
        df.loc[idx, ["open", "high", "low", "close"]] = [100.5, 104.0, 100.0, 101.0]


class ZoneActivationTests(unittest.TestCase):
    def test_zone_activates_only_after_all_confirmation_bars_close(self):
        df = market_frame(70)
        add_demand_candidate(df, 20)

        demand, _ = detect_zones(
            df,
            lookback=100,
            zone_size=2,
            departure_bars=4,
            min_width_atr=0.0,
            max_width_atr=100.0,
            departure_atr_mult=0.1,
        )

        zone = next(z for z in demand if z["formation_idx"] == 20)
        self.assertEqual(24, zone["activation_idx"])
        self.assertEqual(df.iloc[20]["time"], zone["created_at"])
        self.assertEqual(df.iloc[24]["time"], zone["detected_at"])
        self.assertEqual(df.iloc[24]["time"], zone["activated_at"])

    def test_confirmation_bar_overlap_is_not_counted_as_a_touch(self):
        df = market_frame(70)
        add_demand_candidate(df, 20)
        # This overlaps the future zone but occurs before the zone is knowable.
        df.loc[22, ["open", "high", "low", "close"]] = [100.0, 104.0, 95.5, 101.0]

        demand, _ = detect_zones(
            df,
            lookback=100,
            zone_size=2,
            departure_bars=4,
            min_width_atr=0.0,
            max_width_atr=100.0,
            departure_atr_mult=0.1,
        )

        zone = next(z for z in demand if z["formation_idx"] == 20)
        self.assertEqual(0, zone["touches"])
        self.assertEqual([], zone["touch_events"])

    def test_first_post_activation_overlap_is_counted(self):
        df = market_frame(70)
        add_demand_candidate(df, 20)
        df.loc[25, ["open", "high", "low", "close"]] = [100.0, 101.0, 95.5, 100.5]
        # Equal adjacent lows make this a touch event, not a newly confirmed
        # pivot that would later merge into the original zone.
        df.loc[26, ["open", "high", "low", "close"]] = [100.0, 101.0, 95.5, 100.5]

        demand, _ = detect_zones(
            df,
            lookback=100,
            zone_size=2,
            departure_bars=4,
            min_width_atr=0.0,
            max_width_atr=100.0,
            departure_atr_mult=0.1,
        )

        zone = next(z for z in demand if z["formation_idx"] == 20)
        self.assertEqual(1, zone["touches"])
        self.assertEqual(25, zone["touch_events"][0]["start_idx"])

    def test_event_study_does_not_admit_pre_warmup_survivors(self):
        df = market_frame(100)
        add_demand_candidate(df, 20, low=99.0)  # activated before warmup
        add_demand_candidate(df, 65, low=95.0)  # activated at index 69
        # A genuine post-activation touch for the second candidate.
        df.loc[72, ["open", "high", "low", "close"]] = [100.0, 101.0, 95.5, 100.5]
        df.loc[73, ["open", "high", "low", "close"]] = [101.0, 104.0, 95.5, 103.0]

        cfg = {
            "detector_history_bars": 600,
            "warmup_bars": 60,
            "reaction_horizon_bars": 2,
            "lookback": 100,
            "zone_size": 2,
        }
        zones, _ = run_symbol_study("TEST", df, cfg)

        self.assertFalse(zones.empty)
        self.assertTrue((zones["activation_idx"] >= 60).all())
        self.assertNotIn(20, zones["formation_idx"].tolist())
        later = zones[zones["formation_idx"] == 65].iloc[0]
        self.assertEqual(69, later["detection_idx"])
        self.assertEqual(69, later["activation_idx"])
        self.assertEqual(later["detection_time"], later["activated_at"])


if __name__ == "__main__":
    unittest.main()
