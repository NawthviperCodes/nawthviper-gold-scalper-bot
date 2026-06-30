"""Canonical strategy implementations and compatibility adapters."""

from .zone_reversal import ZoneReversalConfig, ZoneReversalStrategy, wilder_atr
from .pipeline import (
    PIPELINE_MODES,
    ParityReport,
    build_zone_reversal_snapshot,
    closed_bars_frame,
    closed_bars_slice,
    normalize_pipeline_mode,
    quote_rejection_reason,
    run_zone_reversal_pipeline,
    timeframe_close_delta,
)

__all__ = [
    "PIPELINE_MODES",
    "ParityReport",
    "ZoneReversalConfig",
    "ZoneReversalStrategy",
    "build_zone_reversal_snapshot",
    "closed_bars_frame",
    "closed_bars_slice",
    "normalize_pipeline_mode",
    "quote_rejection_reason",
    "run_zone_reversal_pipeline",
    "timeframe_close_delta",
    "wilder_atr",
]
