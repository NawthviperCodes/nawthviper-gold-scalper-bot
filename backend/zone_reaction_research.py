import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULTS = {
    "zone_results_dir": "./zone_research_results",
    "output_dir": "./reaction_research_results",
    "symbols": ["XAUUSDz", "EURUSDz", "USDJPYz"],
}


def load_runtime_config(args) -> dict:
    cfg = DEFAULTS.copy()

    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cfg["symbols"] = raw.get("BotSettings", {}).get("SYMBOLS", cfg["symbols"])
        except Exception:
            pass

    if args.zone_results_dir:
        cfg["zone_results_dir"] = args.zone_results_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.symbols:
        cfg["symbols"] = [s.strip() for s in args.symbols.split(",") if s.strip()]

    return cfg


REVERSE_OUTCOMES = {"immediate_reversal", "consolidation_then_reversal"}
BREAKOUT_OUTCOMES = {"breakout"}
NEUTRAL_OUTCOMES = {"no_clear_reaction"}
UNTOUCHED_OUTCOMES = {"untouched", "invalidated_before_touch"}


def load_zones_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # best-effort datetime parsing
    for col in ["zone_time", "first_touch_time", "invalidated_time", "reaction_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def derive_reaction_family(outcome: str) -> str:
    outcome = str(outcome)
    if outcome in REVERSE_OUTCOMES:
        return "reversal"
    if outcome in BREAKOUT_OUTCOMES:
        return "breakout"
    if outcome in NEUTRAL_OUTCOMES:
        return "no_clear_reaction"
    if outcome in UNTOUCHED_OUTCOMES:
        return "untouched_or_invalidated"
    return "other"


def derive_freshness_class(touches) -> str:
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


def safe_rate(num: float, den: float) -> float:
    return round((num / den) * 100, 2) if den else 0.0


def build_symbol_summary(symbol: str, df: pd.DataFrame) -> dict:
    total = len(df)
    touched_df = df[df["touched"] == True].copy() if "touched" in df.columns else df.copy()
    touched = len(touched_df)

    reversal_count = int(touched_df["reaction_family"].eq("reversal").sum())
    breakout_count = int(touched_df["reaction_family"].eq("breakout").sum())
    no_clear_count = int(touched_df["reaction_family"].eq("no_clear_reaction").sum())
    immediate_count = int(touched_df["reaction_outcome"].eq("immediate_reversal").sum())
    consolid_count = int(touched_df["reaction_outcome"].eq("consolidation_then_reversal").sum())

    return {
        "symbol": symbol,
        "total_zones": int(total),
        "touched_zones": int(touched),
        "touched_pct": safe_rate(touched, total),
        "reversal_count": reversal_count,
        "breakout_count": breakout_count,
        "no_clear_count": no_clear_count,
        "immediate_reversal_count": immediate_count,
        "consolidation_reversal_count": consolid_count,
        "reversal_rate_touched_pct": safe_rate(reversal_count, touched),
        "breakout_rate_touched_pct": safe_rate(breakout_count, touched),
        "no_clear_rate_touched_pct": safe_rate(no_clear_count, touched),
        "immediate_reversal_rate_touched_pct": safe_rate(immediate_count, touched),
        "consolidation_reversal_rate_touched_pct": safe_rate(consolid_count, touched),
        "avg_width_atr_ratio": round(float(pd.to_numeric(df.get("width_atr_ratio"), errors="coerce").dropna().mean()), 4) if "width_atr_ratio" in df.columns else np.nan,
        "avg_departure_strength_atr": round(float(pd.to_numeric(df.get("departure_strength_atr"), errors="coerce").dropna().mean()), 4) if "departure_strength_atr" in df.columns else np.nan,
        "avg_touches": round(float(pd.to_numeric(df.get("touches"), errors="coerce").dropna().mean()), 2) if "touches" in df.columns else np.nan,
    }


def crosstab_percent(df: pd.DataFrame, group_col: str, outcome_col: str = "reaction_outcome") -> pd.DataFrame:
    tmp = df.copy()
    tmp[group_col] = tmp[group_col].fillna("unknown") if group_col in tmp.columns else "unknown"
    ct = pd.crosstab(tmp[group_col], tmp[outcome_col], dropna=False)
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    pct = pct.reset_index()
    return pct.round(2)


def grouped_reaction_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows: List[dict] = []
    for grp, g in df.groupby(group_col, dropna=False):
        touched = len(g)
        reversal = int(g["reaction_family"].eq("reversal").sum())
        breakout = int(g["reaction_family"].eq("breakout").sum())
        no_clear = int(g["reaction_family"].eq("no_clear_reaction").sum())
        immediate = int(g["reaction_outcome"].eq("immediate_reversal").sum())
        consolid = int(g["reaction_outcome"].eq("consolidation_then_reversal").sum())
        rows.append({
            group_col: grp if pd.notna(grp) else "unknown",
            "count": touched,
            "reversal_rate_pct": safe_rate(reversal, touched),
            "breakout_rate_pct": safe_rate(breakout, touched),
            "no_clear_rate_pct": safe_rate(no_clear, touched),
            "immediate_reversal_rate_pct": safe_rate(immediate, touched),
            "consolidation_reversal_rate_pct": safe_rate(consolid, touched),
            "avg_width_atr_ratio": round(float(pd.to_numeric(g.get("width_atr_ratio"), errors="coerce").dropna().mean()), 4) if "width_atr_ratio" in g.columns else np.nan,
            "avg_departure_strength_atr": round(float(pd.to_numeric(g.get("departure_strength_atr"), errors="coerce").dropna().mean()), 4) if "departure_strength_atr" in g.columns else np.nan,
            "avg_touches": round(float(pd.to_numeric(g.get("touches"), errors="coerce").dropna().mean()), 2) if "touches" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def run_symbol_reaction_study(symbol: str, df: pd.DataFrame, output_dir: str) -> dict:
    df = df.copy()
    if "reaction_outcome" not in df.columns:
        raise ValueError(f"{symbol} zones file does not contain 'reaction_outcome'")

    df["reaction_family"] = df["reaction_outcome"].apply(derive_reaction_family)
    if "touches" in df.columns:
        df["freshness_class"] = df["touches"].apply(derive_freshness_class)
    else:
        df["freshness_class"] = "unknown"

    touched_df = df[df["touched"] == True].copy() if "touched" in df.columns else df.copy()

    summary = build_symbol_summary(symbol, df)

    # Main grouped analyses on touched zones only
    by_zone_type = grouped_reaction_rates(touched_df, "zone_type") if "zone_type" in touched_df.columns else pd.DataFrame()
    by_freshness = grouped_reaction_rates(touched_df, "freshness_class")
    by_quality = grouped_reaction_rates(touched_df, "quality_bucket") if "quality_bucket" in touched_df.columns else pd.DataFrame()
    by_touch_depth = grouped_reaction_rates(touched_df, "first_touch_depth") if "first_touch_depth" in touched_df.columns else pd.DataFrame()

    # Outcome composition tables
    freshness_outcome_pct = crosstab_percent(touched_df, "freshness_class")
    quality_outcome_pct = crosstab_percent(touched_df, "quality_bucket") if "quality_bucket" in touched_df.columns else pd.DataFrame()
    touch_depth_outcome_pct = crosstab_percent(touched_df, "first_touch_depth") if "first_touch_depth" in touched_df.columns else pd.DataFrame()
    zone_type_outcome_pct = crosstab_percent(touched_df, "zone_type") if "zone_type" in touched_df.columns else pd.DataFrame()

    # Save per-symbol outputs
    base = os.path.join(output_dir, symbol)
    os.makedirs(base, exist_ok=True)

    df.to_csv(os.path.join(base, f"{symbol}_zones_enriched.csv"), index=False)
    by_zone_type.to_csv(os.path.join(base, f"{symbol}_by_zone_type.csv"), index=False)
    by_freshness.to_csv(os.path.join(base, f"{symbol}_by_freshness.csv"), index=False)
    by_quality.to_csv(os.path.join(base, f"{symbol}_by_quality.csv"), index=False)
    by_touch_depth.to_csv(os.path.join(base, f"{symbol}_by_touch_depth.csv"), index=False)
    freshness_outcome_pct.to_csv(os.path.join(base, f"{symbol}_freshness_outcome_pct.csv"), index=False)
    quality_outcome_pct.to_csv(os.path.join(base, f"{symbol}_quality_outcome_pct.csv"), index=False)
    touch_depth_outcome_pct.to_csv(os.path.join(base, f"{symbol}_touch_depth_outcome_pct.csv"), index=False)
    zone_type_outcome_pct.to_csv(os.path.join(base, f"{symbol}_zone_type_outcome_pct.csv"), index=False)

    with open(os.path.join(base, f"{symbol}_reaction_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    print(f"  Zones:                {summary['total_zones']}")
    print(f"  Touched zones:        {summary['touched_zones']} ({summary['touched_pct']}%)")
    print(f"  Reversal rate:        {summary['reversal_rate_touched_pct']}%")
    print(f"  Immediate reversal:   {summary['immediate_reversal_rate_touched_pct']}%")
    print(f"  Consolidation rev.:   {summary['consolidation_reversal_rate_touched_pct']}%")
    print(f"  Breakout rate:        {summary['breakout_rate_touched_pct']}%")
    print(f"  No-clear rate:        {summary['no_clear_rate_touched_pct']}%")

    return {
        "summary": summary,
        "by_zone_type": by_zone_type,
        "by_freshness": by_freshness,
        "by_quality": by_quality,
        "by_touch_depth": by_touch_depth,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 reaction research using Phase 1 zone-study outputs.")
    parser.add_argument("--config", default="config.json", help="Optional config file to read symbols from")
    parser.add_argument("--zone-results-dir", default=None, help="Folder containing *_zones.csv from zone_event_study.py")
    parser.add_argument("--output-dir", default=None, help="Folder to save reaction-research results")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols override")
    args = parser.parse_args()

    cfg = load_runtime_config(args)
    zone_results_dir = cfg["zone_results_dir"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ReactionStudy] Zone results dir: {zone_results_dir}")
    print(f"[ReactionStudy] Output dir:       {output_dir}")
    print(f"[ReactionStudy] Symbols:          {cfg['symbols']}")

    master_rows: List[dict] = []

    for symbol in cfg["symbols"]:
        print(f"\n{'─'*55}\n  Reaction Study: {symbol}\n{'─'*55}")
        path = os.path.join(zone_results_dir, f"{symbol}_zones.csv")
        if not os.path.exists(path):
            print(f"  [!] Missing zones file: {path}")
            continue

        df = load_zones_csv(path)
        res = run_symbol_reaction_study(symbol, df, output_dir)
        master_rows.append(res["summary"])

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_df.to_csv(os.path.join(output_dir, "master_reaction_summary.csv"), index=False)
        with open(os.path.join(output_dir, "master_reaction_summary.json"), "w") as f:
            json.dump(master_rows, f, indent=4, default=str)
        print(f"\n[ReactionStudy] Saved master summary to {output_dir}")
    else:
        print("\n[ReactionStudy] No results produced.")


if __name__ == "__main__":
    main()
