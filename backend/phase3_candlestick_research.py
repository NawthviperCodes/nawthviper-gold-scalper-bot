
import os
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "data_dir": "./backtest_data",
    "zone_results_dir": "./zone_research_results",
    "output_dir": "./candlestick_research_results",
    "symbols": ["XAUUSDz", "EURUSDz", "USDJPYz"],
    "confirm_timeframe": "M5",  # study confirmation candles on M5 by default
    "lookback_candles": 3,
    "touch_window_minutes": 60,  # first_touch_time is currently H1-based
}


# -------------------------------
# Robust loaders
# -------------------------------

def load_mt5_csv(path: str) -> pd.DataFrame:
    """
    Load MT5-exported CSVs:
    - tab or comma separated
    - <DATE>, <TIME>, <OPEN>...
    - plain date/time columns
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").strip().lower() for c in df.columns]

    rename_map = {
        "tickvol": "tick_volume",
        "tick_vol": "tick_volume",
        "vol": "real_volume",
        "volume": "real_volume",
    }
    df.rename(columns=rename_map, inplace=True)

    if "date" in df.columns:
        date_str = df["date"].astype(str).str.replace(".", "-", regex=False)
        if "time" in df.columns:
            time_str = df["time"].astype(str).str.strip()
            df["time"] = pd.to_datetime(
                date_str + " " + time_str,
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
        else:
            df["time"] = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"].astype(str).str.replace(".", "-", regex=False),
            errors="coerce",
        )
    else:
        raise ValueError(f"Cannot find date/time column in {path}. Columns: {list(df.columns)}")

    required = ["time", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Columns: {list(df.columns)}")

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def load_zone_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # best-effort datetime parsing
    for col in [
        "zone_time", "created_at", "detection_time", "activated_at",
        "first_touch_time", "touch_observed_at", "invalidated_time",
        "reaction_bar_time", "reaction_evaluation_start_time", "reaction_time",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_runtime_config(args) -> dict:
    cfg = DEFAULT_CONFIG.copy()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cfg["symbols"] = raw.get("BotSettings", {}).get("SYMBOLS", cfg["symbols"])

            strat = raw.get("StrategyParameters", {})
            tf_confirm = strat.get("TIMEFRAME_CONFIRM", "TIMEFRAME_M5")
            tf_map = {
                "TIMEFRAME_M1": "M1",
                "TIMEFRAME_M5": "M5",
                "TIMEFRAME_M15": "M15",
                "TIMEFRAME_H1": "H1",
                "TIMEFRAME_H4": "H4",
            }
            cfg["confirm_timeframe"] = tf_map.get(tf_confirm, cfg["confirm_timeframe"])

    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.zone_results_dir:
        cfg["zone_results_dir"] = args.zone_results_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.symbols:
        cfg["symbols"] = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.timeframe:
        cfg["confirm_timeframe"] = args.timeframe

    return cfg


# -------------------------------
# Candlestick helpers
# -------------------------------

def body(c: pd.Series) -> float:
    return abs(float(c["close"]) - float(c["open"]))

def candle_range(c: pd.Series) -> float:
    return max(float(c["high"]) - float(c["low"]), 1e-9)

def upper_wick(c: pd.Series) -> float:
    return float(c["high"]) - max(float(c["open"]), float(c["close"]))

def lower_wick(c: pd.Series) -> float:
    return min(float(c["open"]), float(c["close"])) - float(c["low"])

def is_bullish(c: pd.Series) -> bool:
    return float(c["close"]) > float(c["open"])

def is_bearish(c: pd.Series) -> bool:
    return float(c["close"]) < float(c["open"])


def is_bullish_engulfing(c1: pd.Series, c2: pd.Series) -> bool:
    return (
        is_bearish(c1)
        and is_bullish(c2)
        and float(c2["open"]) <= float(c1["close"])
        and float(c2["close"]) >= float(c1["open"])
        and body(c2) > body(c1) * 0.8
    )

def is_bearish_engulfing(c1: pd.Series, c2: pd.Series) -> bool:
    return (
        is_bullish(c1)
        and is_bearish(c2)
        and float(c2["open"]) >= float(c1["close"])
        and float(c2["close"]) <= float(c1["open"])
        and body(c2) > body(c1) * 0.8
    )

def is_bullish_pin_bar(c: pd.Series) -> bool:
    return lower_wick(c) >= body(c) * 2.0 and upper_wick(c) <= body(c) * 1.2 and is_bullish(c)

def is_bearish_pin_bar(c: pd.Series) -> bool:
    return upper_wick(c) >= body(c) * 2.0 and lower_wick(c) <= body(c) * 1.2 and is_bearish(c)

def is_doji(c: pd.Series) -> bool:
    return body(c) <= candle_range(c) * 0.1

def is_inside_bar(c1: pd.Series, c2: pd.Series) -> bool:
    return float(c2["high"]) <= float(c1["high"]) and float(c2["low"]) >= float(c1["low"])

def is_bullish_morning_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    mid_c1 = (float(c1["open"]) + float(c1["close"])) / 2
    return (
        is_bearish(c1)
        and body(c2) <= body(c1) * 0.6
        and is_bullish(c3)
        and float(c3["close"]) > mid_c1
    )

def is_bearish_evening_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    mid_c1 = (float(c1["open"]) + float(c1["close"])) / 2
    return (
        is_bullish(c1)
        and body(c2) <= body(c1) * 0.6
        and is_bearish(c3)
        and float(c3["close"]) < mid_c1
    )


def classify_pattern(candles: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns:
      pattern_name, pattern_side ('buy'/'sell'/'neutral'/'none')
    Uses the last 1-3 candles ending at the touch trigger candle.
    """
    if len(candles) < 1:
        return "no_pattern", "none"

    c3 = candles.iloc[-1]
    c2 = candles.iloc[-2] if len(candles) >= 2 else None
    c1 = candles.iloc[-3] if len(candles) >= 3 else None

    if c2 is not None:
        if is_bullish_engulfing(c2, c3):
            return "bullish_engulfing", "buy"
        if is_bearish_engulfing(c2, c3):
            return "bearish_engulfing", "sell"
        if is_inside_bar(c2, c3):
            return "inside_bar", "neutral"

    if is_bullish_pin_bar(c3):
        return "bullish_pin_bar", "buy"
    if is_bearish_pin_bar(c3):
        return "bearish_pin_bar", "sell"
    if is_doji(c3):
        return "doji", "neutral"

    if c1 is not None and c2 is not None:
        if is_bullish_morning_star(c1, c2, c3):
            return "morning_star", "buy"
        if is_bearish_evening_star(c1, c2, c3):
            return "evening_star", "sell"

    return "no_pattern", "none"


def desired_side_from_zone(zone_type: str) -> str:
    return "buy" if zone_type == "demand" else "sell"


def freshness_class(touches: float) -> str:
    try:
        t = int(touches)
    except Exception:
        return "unknown"
    if t <= 0:
        return "fresh"
    if t == 1:
        return "first_retest"
    if t == 2:
        return "second_retest"
    return "multi_retest"


def safe_pct_table(df: pd.DataFrame, row_col: str, outcome_col: str = "reaction_outcome") -> pd.DataFrame:
    if df.empty or row_col not in df.columns:
        return pd.DataFrame()
    counts = pd.crosstab(df[row_col], df[outcome_col], dropna=False)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return pct.round(2)


def infer_bar_interval(df: pd.DataFrame) -> Optional[pd.Timedelta]:
    if df is None or len(df) < 2 or "time" not in df.columns:
        return None
    deltas = pd.Series(pd.to_datetime(df["time"])).diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    return deltas.median() if not deltas.empty else None


def find_touch_m5_index(
    m5_df: pd.DataFrame,
    first_touch_time: pd.Timestamp,
    zone_bottom: float,
    zone_top: float,
    touch_observed_at: Optional[pd.Timestamp] = None,
    touch_window_minutes: int = 60,
) -> Optional[int]:
    """Find the first fully closed M5 candle that actually overlaps the zone.

    ``first_touch_time`` is the H1 touch bar's open timestamp and
    ``touch_observed_at`` is when that H1 bar closed.  A candidate M5 candle
    must open inside that interval, close no later than the H1 observation,
    and genuinely overlap the zone.  There is intentionally no nearest-candle
    fallback because inventing a touch changes the researched population.
    """
    if pd.isna(first_touch_time):
        return None

    first_touch_time = pd.Timestamp(first_touch_time)
    if touch_observed_at is None or pd.isna(touch_observed_at):
        touch_observed_at = first_touch_time + pd.Timedelta(minutes=touch_window_minutes)
    else:
        touch_observed_at = pd.Timestamp(touch_observed_at)
    if touch_observed_at <= first_touch_time:
        return None

    interval = infer_bar_interval(m5_df)
    if interval is None:
        return None

    m5_open = pd.to_datetime(m5_df["time"])
    m5_close = m5_open + interval
    window = m5_df[
        (m5_open >= first_touch_time)
        & (m5_open < touch_observed_at)
        & (m5_close <= touch_observed_at)
    ]
    if window.empty:
        return None

    overlap = window[(window["high"] >= zone_bottom) & (window["low"] <= zone_top)]
    return int(overlap.index[0]) if not overlap.empty else None


def summarize_symbol(enriched: pd.DataFrame) -> dict:
    touched = enriched[enriched["touched"] == True].copy()
    if touched.empty:
        return {
            "zones": int(len(enriched)),
            "touched_zones": 0,
            "evaluable_touched_zones": 0,
            "right_censored_count": 0,
            "pattern_evaluable_touches": 0,
            "reversal_rate_pct": 0.0,
            "immediate_reversal_pct": 0.0,
            "consolidation_then_reversal_pct": 0.0,
            "breakout_pct": 0.0,
            "no_clear_pct": 0.0,
            "pattern_counts": {},
            "pattern_vs_outcome": {},
        }

    censored = touched["reaction_outcome"].eq("right_censored")
    evaluable = touched[~censored].copy()
    pattern_evaluable = evaluable[evaluable["m5_alignment_status"].eq("m5_touch_aligned")].copy()

    reversal_mask = evaluable["reaction_outcome"].isin(["immediate_reversal", "consolidation_then_reversal"])
    immediate_mask = evaluable["reaction_outcome"].eq("immediate_reversal")
    consol_mask = evaluable["reaction_outcome"].eq("consolidation_then_reversal")
    breakout_mask = evaluable["reaction_outcome"].eq("breakout")
    no_clear_mask = evaluable["reaction_outcome"].eq("no_clear_reaction")

    pattern_counts = pattern_evaluable["pattern_name"].value_counts(dropna=False).to_dict()
    pattern_vs_outcome = pd.crosstab(
        pattern_evaluable["pattern_name"], pattern_evaluable["reaction_outcome"], dropna=False
    ).to_dict()

    return {
        "zones": int(len(enriched)),
        "touched_zones": int(len(touched)),
        "evaluable_touched_zones": int(len(evaluable)),
        "right_censored_count": int(censored.sum()),
        "pattern_evaluable_touches": int(len(pattern_evaluable)),
        "m5_alignment_failure_count": int(len(evaluable) - len(pattern_evaluable)),
        "reversal_rate_pct": round(float(reversal_mask.mean()) * 100, 2) if len(evaluable) else 0.0,
        "immediate_reversal_pct": round(float(immediate_mask.mean()) * 100, 2) if len(evaluable) else 0.0,
        "consolidation_then_reversal_pct": round(float(consol_mask.mean()) * 100, 2) if len(evaluable) else 0.0,
        "breakout_pct": round(float(breakout_mask.mean()) * 100, 2) if len(evaluable) else 0.0,
        "no_clear_pct": round(float(no_clear_mask.mean()) * 100, 2) if len(evaluable) else 0.0,
        "pattern_counts": pattern_counts,
        "pattern_vs_outcome": pattern_vs_outcome,
    }


def enrich_symbol(symbol: str, zone_df: pd.DataFrame, m5_df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    if zone_df.empty:
        return zone_df.copy(), summarize_symbol(zone_df)

    rows = []
    touch_window_minutes = int(cfg["touch_window_minutes"])
    m5_interval = infer_bar_interval(m5_df)

    for _, row in zone_df.iterrows():
        rec = row.to_dict()

        rec["freshness_class"] = freshness_class(rec.get("touches", 0))
        rec["pattern_name"] = "no_pattern"
        rec["pattern_side"] = "none"
        rec["pattern_alignment"] = "none"
        rec["m5_alignment_status"] = "not_evaluated"
        rec["m5_touch_idx"] = None
        rec["m5_touch_time"] = pd.NaT
        rec["pattern_knowledge_time"] = pd.NaT
        rec["touch_observation_boundary"] = pd.NaT
        rec["touch_boundary_source"] = "none"

        if str(rec.get("reaction_outcome", "")) == "right_censored":
            rec["m5_alignment_status"] = "excluded_right_censored"
            rows.append(rec)
            continue

        first_touch_time = rec.get("first_touch_time", pd.NaT)
        touch_observed_at = rec.get("touch_observed_at", pd.NaT)
        if pd.isna(first_touch_time):
            rec["m5_alignment_status"] = "missing_h1_touch_time"
            rows.append(rec)
            continue
        if pd.isna(touch_observed_at):
            observation_boundary = pd.Timestamp(first_touch_time) + pd.Timedelta(minutes=touch_window_minutes)
            rec["touch_boundary_source"] = "legacy_inferred"
        else:
            observation_boundary = pd.Timestamp(touch_observed_at)
            rec["touch_boundary_source"] = "explicit_touch_observed_at"
        rec["touch_observation_boundary"] = observation_boundary
        zone_bottom = float(rec.get("zone_bottom"))
        zone_top = float(rec.get("zone_top"))
        zone_type = rec.get("zone_type", "")
        desired_side = desired_side_from_zone(zone_type) if zone_type in ("demand", "supply") else "none"

        touch_idx = find_touch_m5_index(
            m5_df,
            first_touch_time,
            zone_bottom,
            zone_top,
            touch_observed_at=observation_boundary,
            touch_window_minutes=touch_window_minutes,
        )

        if touch_idx is not None and touch_idx >= 0:
            rec["m5_alignment_status"] = "m5_touch_aligned"
            rec["m5_touch_idx"] = int(touch_idx)
            rec["m5_touch_time"] = pd.Timestamp(m5_df.loc[touch_idx, "time"])
            if m5_interval is not None:
                rec["pattern_knowledge_time"] = rec["m5_touch_time"] + m5_interval
                if rec["pattern_knowledge_time"] > observation_boundary:
                    raise AssertionError("M5 pattern uses a candle that closed after the H1 touch observation")
            start_idx = max(0, touch_idx - (cfg["lookback_candles"] - 1))
            candles = m5_df.iloc[start_idx:touch_idx + 1].copy()

            if len(candles) >= 1:
                pattern_name, pattern_side = classify_pattern(candles)
                rec["pattern_name"] = pattern_name
                rec["pattern_side"] = pattern_side

                if pattern_side == "none":
                    rec["pattern_alignment"] = "none"
                elif pattern_side == "neutral":
                    rec["pattern_alignment"] = "neutral"
                elif pattern_side == desired_side:
                    rec["pattern_alignment"] = "aligned"
                else:
                    rec["pattern_alignment"] = "conflict"
        else:
            rec["m5_alignment_status"] = "no_m5_zone_overlap"

        rows.append(rec)

    enriched = pd.DataFrame(rows)
    summary = summarize_symbol(enriched)
    return enriched, summary


def save_symbol_outputs(symbol: str, enriched: pd.DataFrame, summary: dict, output_dir: str):
    sym_dir = os.path.join(output_dir, symbol)
    os.makedirs(sym_dir, exist_ok=True)

    enriched.to_csv(os.path.join(sym_dir, f"{symbol}_zones_enriched.csv"), index=False)

    with open(os.path.join(sym_dir, f"{symbol}_reaction_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    evaluable = enriched[
        (enriched["touched"] == True)
        & ~enriched["reaction_outcome"].eq("right_censored")
    ].copy()
    pattern_evaluable = evaluable[evaluable["m5_alignment_status"].eq("m5_touch_aligned")].copy()

    # Counts.  Structural zone tables use all evaluable touches; pattern
    # tables additionally require a genuine, causally aligned M5 overlap.
    for col, suffix in [
        ("freshness_class", "by_freshness"),
        ("quality_bucket", "by_quality"),
        ("first_touch_depth", "by_touch_depth"),
        ("zone_type", "by_zone_type"),
        ("pattern_name", "by_pattern"),
        ("pattern_alignment", "by_pattern_alignment"),
    ]:
        source = pattern_evaluable if col in {"pattern_name", "pattern_alignment"} else evaluable
        if col in source.columns:
            ctab = pd.crosstab(source[col], source["reaction_outcome"], dropna=False)
            ctab.to_csv(os.path.join(sym_dir, f"{symbol}_{suffix}.csv"))

            pct = safe_pct_table(source, col)
            pct.to_csv(os.path.join(sym_dir, f"{symbol}_{suffix}_outcome_pct.csv"))

    # Simple aligned vs no/aligned/conflict summary
    if "pattern_alignment" in pattern_evaluable.columns:
        pa = pd.crosstab(pattern_evaluable["pattern_alignment"], pattern_evaluable["reaction_outcome"], dropna=False)
        pa.to_csv(os.path.join(sym_dir, f"{symbol}_pattern_alignment.csv"))
        pa_pct = safe_pct_table(pattern_evaluable, "pattern_alignment")
        pa_pct.to_csv(os.path.join(sym_dir, f"{symbol}_pattern_alignment_outcome_pct.csv"))


def main():
    parser = argparse.ArgumentParser(description="Phase 3 candlestick research at zone touches.")
    parser.add_argument("--config", default="config.json", help="Optional config to read symbols / confirm TF")
    parser.add_argument("--data-dir", default=None, help="Folder containing historical CSVs (same backtest data folder)")
    parser.add_argument("--zone-results-dir", default=None, help="Folder containing zone_event_study outputs")
    parser.add_argument("--output-dir", default=None, help="Folder to save candlestick research outputs")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols override")
    parser.add_argument("--timeframe", default=None, help="Override confirmation timeframe (e.g. M5)")
    args = parser.parse_args()

    cfg = load_runtime_config(args)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"[CandleStudy] Data dir:         {cfg['data_dir']}")
    print(f"[CandleStudy] Zone results dir: {cfg['zone_results_dir']}")
    print(f"[CandleStudy] Output dir:       {cfg['output_dir']}")
    print(f"[CandleStudy] Symbols:          {cfg['symbols']}")
    print(f"[CandleStudy] Confirm TF:       {cfg['confirm_timeframe']}")

    master_rows = []

    for symbol in cfg["symbols"]:
        print(f"\n{'─'*55}\n  Candlestick Study: {symbol}\n{'─'*55}")

        zone_path = os.path.join(cfg["zone_results_dir"], f"{symbol}_zones.csv")
        tf = cfg["confirm_timeframe"]
        price_path = os.path.join(cfg["data_dir"], f"{symbol}_{tf}.csv")

        if not os.path.exists(zone_path):
            print(f"  [!] Missing zone file: {zone_path}")
            continue
        if not os.path.exists(price_path):
            print(f"  [!] Missing price file: {price_path}")
            continue

        zone_df = load_zone_csv(zone_path)
        price_df = load_mt5_csv(price_path)

        enriched, summary = enrich_symbol(symbol, zone_df, price_df, cfg)
        save_symbol_outputs(symbol, enriched, summary, cfg["output_dir"])

        print(f"  Zones:                  {summary['zones']}")
        print(f"  Touched zones:          {summary['touched_zones']}")
        print(f"  Evaluable touches:      {summary['evaluable_touched_zones']}")
        print(f"  Right-censored:         {summary['right_censored_count']}")
        print(f"  M5 pattern-aligned:     {summary['pattern_evaluable_touches']}")
        print(f"  M5 alignment failures:  {summary['m5_alignment_failure_count']}")
        print(f"  Reversal rate:          {summary['reversal_rate_pct']}%")
        print(f"  Immediate reversal:     {summary['immediate_reversal_pct']}%")
        print(f"  Consolidation reversal: {summary['consolidation_then_reversal_pct']}%")
        print(f"  Breakout rate:          {summary['breakout_pct']}%")
        print(f"  No-clear rate:          {summary['no_clear_pct']}%")
        print(f"  Pattern counts:         {summary['pattern_counts']}")

        master_rows.append({
            "symbol": symbol,
            **summary
        })

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_df.to_csv(os.path.join(cfg["output_dir"], "master_candlestick_summary.csv"), index=False)
        with open(os.path.join(cfg["output_dir"], "master_candlestick_summary.json"), "w") as f:
            json.dump(master_rows, f, indent=4, default=str)
        print(f"\n[CandleStudy] Saved master summary to {cfg['output_dir']}")
    else:
        print("\n[CandleStudy] No symbol results were produced.")


if __name__ == "__main__":
    main()
