import unittest

import pandas as pd

from trade_decision_engine import (
    REJECTION_STATS,
    _build_signal,
    run_trade_decision_engine,
)


def reclaim_bars():
    return pd.DataFrame(
        [
            {"open": 101.2, "high": 101.8, "low": 100.8, "close": 101.0},
            {"open": 101.0, "high": 102.6, "low": 100.5, "close": 102.5},
        ]
    )


def zone(quality="A"):
    return {
        "type": "demand",
        "price": 100.0,
        "bottom": 100.0,
        "top": 102.0,
        "quality_bucket": quality,
        "touches": 0,
        "displacement_origin_low": 99.0,
        "displacement_origin_high": 102.0,
    }


def decide(*, quality="A", bias="NEUTRAL", telemetry=True):
    return run_trade_decision_engine(
        symbol="TEST",
        point=0.01,
        current_price=102.5,
        trend="uptrend",
        demand_zones=[zone(quality)],
        supply_zones=[],
        m5_candles_for_patterns=reclaim_bars(),
        TP_RATIO=2.0,
        atr=1.0,
        m5_context={"trend": "uptrend"},
        htf_bias=bias,
        thresholds={"MAX_TOUCH_ALLOWED": 2, "MIN_RR_FILTER": 1.3},
        strategy_active_override=True,
        emit_telemetry=telemetry,
        spread=0.2,
    )[0]


class TradeDecisionRebuildTests(unittest.TestCase):
    def setUp(self):
        for reason in REJECTION_STATS:
            REJECTION_STATS[reason] = 0

    def test_b_zone_is_hard_rejected(self):
        self.assertEqual([], decide(quality="B"))
        self.assertEqual(1, REJECTION_STATS["B Zone Rejected"])

    def test_opposite_htf_bias_is_hard_rejected(self):
        self.assertEqual([], decide(bias="DOWN"))
        self.assertEqual(1, REJECTION_STATS["HTF Bias Conflict"])

    def test_neutral_htf_bias_passes_and_reason_has_no_pattern(self):
        signals = decide(bias="NEUTRAL")
        self.assertEqual(1, len(signals))
        self.assertEqual("buy_reclaim", signals[0]["reason"])

    def test_buy_stop_uses_origin_and_target_uses_ask_fill(self):
        signal = _build_signal(
            symbol="TEST",
            side="buy",
            current_price=102.5,
            zone=zone(),
            touch_ctx={"sweep_wick": 100.5},
            atr_val=1.0,
            tp_ratio=2.0,
            strategy_name="test",
            reason="buy_reclaim",
            confidence=0.7,
            spread=0.2,
        )
        self.assertAlmostEqual(98.85, signal["sl"])
        fill = 102.7
        self.assertAlmostEqual(fill + (fill - 98.85) * 2.0, signal["tp"])

    def test_structural_stop_falls_back_to_touch_wick(self):
        test_zone = zone()
        test_zone["displacement_origin_low"] = 101.0
        signal = _build_signal(
            symbol="TEST",
            side="buy",
            current_price=102.5,
            zone=test_zone,
            touch_ctx={"sweep_wick": 100.5},
            atr_val=1.0,
            tp_ratio=2.0,
            strategy_name="test",
            reason="buy_reclaim",
            confidence=0.7,
            spread=0.2,
        )
        self.assertAlmostEqual(100.35, signal["sl"])


if __name__ == "__main__":
    unittest.main()
