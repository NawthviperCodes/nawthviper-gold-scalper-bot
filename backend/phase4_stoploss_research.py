import os
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    "data_dir": "./backtest_data",
    "research_results_dir": "./candlestick_research_results",
    "output_dir": "./stoploss_research_results",
    "symbols": ["XAUUSDz", "EURUSDz", "USDJPYz"],
    "zone_timeframe": "H1",
    "reaction_horizon_bars": 12,
    "tp_target_mode": "zone_event_target",  # or fixed_r_multiple
    "fixed_r_multiple": 2.0,
    "wick_buffer_mults": [0.15, 0.25],
    "deep_buffer_mult": 0.25,
}

# -------------------------------
# Robust loaders
# -------------------------------

def load_mt5_csv(path: str) -> pd.DataFrame:
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


def load_research_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["zone_time", "first_touch_time", "invalidated_time", "reaction_time"]:
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
            tf_zone = strat.get("TIMEFRAME_ZONE", "TIMEFRAME_H1")
            tf_map = {
                "TIMEFRAME_M1": "M1",
                "TIMEFRAME_M5": "M5",
                "TIMEFRAME_M15": "M15",
                "TIMEFRAME_H1": "H1",
                "TIMEFRAME_H4": "H4",
            }
            cfg["zone_timeframe"] = tf_map.get(tf_zone, cfg["zone_timeframe"])

    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.research_results_dir:
        cfg["research_results_dir"] = args.research_results_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.symbols:
        cfg["symbols"] = [s.strip() for s in args.symbols.split(",") if s.strip()]
    return cfg

# -------------------------------
# Stop models
# -------------------------------

def reaction_target_distance(row: pd.Series) -> float:
    width = float(row.get("width", row.get("zone_top") - row.get("zone_bottom")))
    atr = float(row.get("atr_at_formation", 0.0) or 0.0)
    return max(width, atr * 0.50, 1e-9)


def make_stop_models(row: pd.Series, touch_candle: pd.Series, cfg: dict) -> Dict[str, float]:
    atr = float(row.get("atr_at_formation", 0.0) or 0.0)
    zone_bottom = float(row["zone_bottom"])
    zone_top = float(row["zone_top"])
    low = float(touch_candle["low"])
    high = float(touch_candle["high"])
    side = "buy" if row["zone_type"] == "demand" else "sell"

    models = {}
    if side == "buy":
        models["zone_edge"] = zone_bottom
        models["sweep_wick"] = low
        models[f"wick_atr_{cfg['wick_buffer_mults'][0]:.2f}"] = low - atr * cfg["wick_buffer_mults"][0]
        models[f"wick_atr_{cfg['wick_buffer_mults'][1]:.2f}"] = low - atr * cfg["wick_buffer_mults"][1]
        models["deep_invalidation"] = min(zone_bottom, low) - atr * cfg["deep_buffer_mult"]
    else:
        models["zone_edge"] = zone_top
        models["sweep_wick"] = high
        models[f"wick_atr_{cfg['wick_buffer_mults'][0]:.2f}"] = high + atr * cfg["wick_buffer_mults"][0]
        models[f"wick_atr_{cfg['wick_buffer_mults'][1]:.2f}"] = high + atr * cfg["wick_buffer_mults"][1]
        models["deep_invalidation"] = max(zone_top, high) + atr * cfg["deep_buffer_mult"]
    return models


def evaluate_stop_model(df: pd.DataFrame, touch_idx: int, row: pd.Series, entry: float, stop: float, horizon: int, cfg: dict) -> Dict[str, object]:
    side = "buy" if row["zone_type"] == "demand" else "sell"
    end_idx = min(len(df) - 1, touch_idx + horizon)
    target_dist = reaction_target_distance(row)

    if side == "buy":
        risk = entry - stop
        if risk <= 0:
            return {"status": "invalid_stop", "r_multiple": np.nan, "false_stop": False, "target_hit": False}
        target = entry + target_dist if cfg["tp_target_mode"] == "zone_event_target" else entry + risk * cfg["fixed_r_multiple"]
    else:
        risk = stop - entry
        if risk <= 0:
            return {"status": "invalid_stop", "r_multiple": np.nan, "false_stop": False, "target_hit": False}
        target = entry - target_dist if cfg["tp_target_mode"] == "zone_event_target" else entry - risk * cfg["fixed_r_multiple"]

    stop_idx = None
    target_idx = None
    ambiguous_idx = None

    for j in range(touch_idx, end_idx + 1):
        c = df.iloc[j]
        low = float(c["low"])
        high = float(c["high"])
        if side == "buy":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:
            ambiguous_idx = j
            break
        if stop_hit:
            stop_idx = j
            break
        if target_hit:
            target_idx = j
            break

    false_stop = False
    status = "no_resolution"
    r_multiple = np.nan
    target_hit_flag = False

    if ambiguous_idx is not None:
        status = "ambiguous_same_bar"
        r_multiple = -1.0  # conservative
    elif stop_idx is not None:
        status = "stopped_before_target"
        r_multiple = -1.0
        # false stop if target is reached later in remaining horizon
        for j in range(stop_idx + 1, end_idx + 1):
            c = df.iloc[j]
            if side == "buy" and float(c["high"]) >= target:
                false_stop = True
                break
            if side == "sell" and float(c["low"]) <= target:
                false_stop = True
                break
    elif target_idx is not None:
        status = "target_before_stop"
        target_hit_flag = True
        if side == "buy":
            r_multiple = (target - entry) / risk
        else:
            r_multiple = (entry - target) / risk

    return {
        "status": status,
        "r_multiple": r_multiple,
        "false_stop": false_stop,
        "target_hit": target_hit_flag,
        "risk_price": risk,
        "target_price": target,
        "stop_price": stop,
    }

# -------------------------------
# Research helpers
# -------------------------------

def choose_source_file(research_dir: str, symbol: str) -> str:
    """
    Phase 3 saves files inside per-symbol folders:
      ./candlestick_research_results/XAUUSDz/XAUUSDz_zones_enriched.csv

    But some earlier scripts may save directly in the root:
      ./candlestick_research_results/XAUUSDz_zones_enriched.csv

    This function checks both layouts.
    """
    candidates = [
        # Phase 3 per-symbol subfolder layout
        os.path.join(research_dir, symbol, f"{symbol}_zones_enriched.csv"),
        os.path.join(research_dir, symbol, f"{symbol}_zones.csv"),

        # Flat root layout fallback
        os.path.join(research_dir, f"{symbol}_zones_enriched.csv"),
        os.path.join(research_dir, f"{symbol}_zones.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find enriched or basic zone file for {symbol} in {research_dir}. "
        f"Checked: {candidates}"
    )


def ensure_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "freshness_class" not in out.columns:
        def freshness_from_touches(v):
            try:
                t = int(v)
            except Exception:
                return "unknown"
            if t <= 0:
                return "fresh"
            if t == 1:
                return "first_retest"
            if t == 2:
                return "second_retest"
            return "multi_retest"
        out["freshness_class"] = out.get("touches", pd.Series(index=out.index)).map(freshness_from_touches)
    if "reaction_family" not in out.columns and "reaction_outcome" in out.columns:
        def fam(x):
            x = str(x)
            if x in ("immediate_reversal", "consolidation_then_reversal"):
                return "reversal"
            if x == "breakout":
                return "breakout"
            if x == "invalidated_before_touch":
                return "invalidated_before_touch"
            return "other"
        out["reaction_family"] = out["reaction_outcome"].map(fam)
    return out


def summarize_stop_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("stop_model"):
        valid = g[g["status"] != "invalid_stop"].copy()
        if valid.empty:
            rows.append({
                "stop_model": model,
                "trades": 0,
                "target_hit_rate_pct": 0.0,
                "stop_before_target_rate_pct": 0.0,
                "false_stop_rate_pct": 0.0,
                "ambiguous_rate_pct": 0.0,
                "avg_r_multiple": np.nan,
                "median_r_multiple": np.nan,
                "avg_risk_atr": np.nan,
            })
            continue
        trades = len(valid)
        target_rate = (valid["status"] == "target_before_stop").mean() * 100.0
        stop_rate = (valid["status"] == "stopped_before_target").mean() * 100.0
        false_stop_rate = valid["false_stop"].mean() * 100.0
        amb_rate = (valid["status"] == "ambiguous_same_bar").mean() * 100.0
        rows.append({
            "stop_model": model,
            "trades": trades,
            "target_hit_rate_pct": round(float(target_rate), 2),
            "stop_before_target_rate_pct": round(float(stop_rate), 2),
            "false_stop_rate_pct": round(float(false_stop_rate), 2),
            "ambiguous_rate_pct": round(float(amb_rate), 2),
            "avg_r_multiple": round(float(valid["r_multiple"].dropna().mean()), 4) if valid["r_multiple"].notna().any() else np.nan,
            "median_r_multiple": round(float(valid["r_multiple"].dropna().median()), 4) if valid["r_multiple"].notna().any() else np.nan,
            "avg_risk_atr": round(float((valid["risk_price"] / valid["atr_at_formation"]).replace([np.inf, -np.inf], np.nan).dropna().mean()), 4) if "atr_at_formation" in valid.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values("stop_model").reset_index(drop=True)


def grouped_stop_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (model, grp), g in df.groupby(["stop_model", group_col], dropna=False):
        valid = g[g["status"] != "invalid_stop"]
        if valid.empty:
            continue
        rows.append({
            "stop_model": model,
            group_col: grp,
            "trades": int(len(valid)),
            "target_hit_rate_pct": round(float((valid["status"] == "target_before_stop").mean() * 100), 2),
            "stop_before_target_rate_pct": round(float((valid["status"] == "stopped_before_target").mean() * 100), 2),
            "false_stop_rate_pct": round(float(valid["false_stop"].mean() * 100), 2),
            "avg_r_multiple": round(float(valid["r_multiple"].dropna().mean()), 4) if valid["r_multiple"].notna().any() else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# -------------------------------
# Core study
# -------------------------------

def run_symbol_study(symbol: str, price_df: pd.DataFrame, zones_df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[dict] = []
    horizon = int(cfg["reaction_horizon_bars"])

    for _, row in zones_df.iterrows():
        if not bool(row.get("touched", False)):
            continue
        ft_idx = row.get("first_touch_idx")
        if pd.isna(ft_idx):
            continue
        touch_idx = int(float(ft_idx))
        if touch_idx < 0 or touch_idx >= len(price_df):
            continue

        touch_candle = price_df.iloc[touch_idx]
        entry = float(touch_candle["close"])
        atr = float(row.get("atr_at_formation", 0.0) or 0.0)

        stop_models = make_stop_models(row, touch_candle, cfg)
        for model_name, stop_price in stop_models.items():
            eval_res = evaluate_stop_model(price_df, touch_idx, row, entry, float(stop_price), horizon, cfg)
            rec = {
                "symbol": symbol,
                "stop_model": model_name,
                "zone_type": row.get("zone_type"),
                "zone_time": row.get("zone_time"),
                "first_touch_time": row.get("first_touch_time"),
                "first_touch_depth": row.get("first_touch_depth", ""),
                "freshness_class": row.get("freshness_class", "unknown"),
                "quality_bucket": row.get("quality_bucket", "unknown"),
                "reaction_outcome": row.get("reaction_outcome", "unknown"),
                "reaction_family": row.get("reaction_family", "unknown"),
                "touches": row.get("touches", np.nan),
                "width": row.get("width", np.nan),
                "width_atr_ratio": row.get("width_atr_ratio", np.nan),
                "departure_strength_atr": row.get("departure_strength_atr", np.nan),
                "atr_at_formation": atr,
                "entry_price": entry,
            }
            rec.update(eval_res)
            records.append(rec)

    results_df = pd.DataFrame(records)
    summary_df = summarize_stop_results(results_df) if not results_df.empty else pd.DataFrame()
    return results_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Phase 4 stop-loss event study")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--research-results-dir", default=None,
                        help="Folder with *_zones_enriched.csv or *_zones.csv (defaults to candlestick_research_results)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--symbols", default=None)
    args = parser.parse_args()

    cfg = load_runtime_config(args)
    data_dir = cfg["data_dir"]
    research_dir = cfg["research_results_dir"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[StopStudy] Data dir:           {data_dir}")
    print(f"[StopStudy] Research dir:       {research_dir}")
    print(f"[StopStudy] Output dir:         {output_dir}")
    print(f"[StopStudy] Symbols:            {cfg['symbols']}")
    print(f"[StopStudy] Zone timeframe:     {cfg['zone_timeframe']}")

    master_summary_rows = []

    for symbol in cfg["symbols"]:
        print(f"\n{'─'*55}\n  Stop-Loss Study: {symbol}\n{'─'*55}")
        price_path = os.path.join(data_dir, f"{symbol}_{cfg['zone_timeframe']}.csv")
        if not os.path.exists(price_path):
            print(f"  [!] Missing price file: {price_path}")
            continue

        source_path = choose_source_file(research_dir, symbol)
        price_df = load_mt5_csv(price_path)
        zones_df = ensure_classes(load_research_csv(source_path))

        results_df, summary_df = run_symbol_study(symbol, price_df, zones_df, cfg)
        if results_df.empty:
            print("  [!] No stop-study results produced.")
            continue

        sym_dir = os.path.join(output_dir, symbol)
        os.makedirs(sym_dir, exist_ok=True)

        results_df.to_csv(os.path.join(sym_dir, f"{symbol}_stop_events.csv"), index=False)
        summary_df.to_csv(os.path.join(sym_dir, f"{symbol}_stop_model_summary.csv"), index=False)

        by_freshness = grouped_stop_summary(results_df, "freshness_class")
        by_quality = grouped_stop_summary(results_df, "quality_bucket")
        by_touch = grouped_stop_summary(results_df, "first_touch_depth")

        if not by_freshness.empty:
            by_freshness.to_csv(os.path.join(sym_dir, f"{symbol}_stop_by_freshness.csv"), index=False)
        if not by_quality.empty:
            by_quality.to_csv(os.path.join(sym_dir, f"{symbol}_stop_by_quality.csv"), index=False)
        if not by_touch.empty:
            by_touch.to_csv(os.path.join(sym_dir, f"{symbol}_stop_by_touch_depth.csv"), index=False)

        summary_json = {
            "symbol": symbol,
            "stop_models": summary_df.to_dict(orient="records"),
        }
        with open(os.path.join(sym_dir, f"{symbol}_stop_summary.json"), "w") as f:
            json.dump(summary_json, f, indent=4, default=str)

        print(f"  Zones used:          {zones_df['touched'].sum() if 'touched' in zones_df.columns else len(zones_df)}")
        for _, r in summary_df.iterrows():
            print(f"  {r['stop_model']:<18} target%={r['target_hit_rate_pct']:>6} stop%={r['stop_before_target_rate_pct']:>6} false_stop%={r['false_stop_rate_pct']:>6} avgR={r['avg_r_multiple']:>7}")
            row = {"symbol": symbol, **r.to_dict()}
            master_summary_rows.append(row)

    if master_summary_rows:
        master_df = pd.DataFrame(master_summary_rows)
        master_df.to_csv(os.path.join(output_dir, "master_stop_model_summary.csv"), index=False)
        with open(os.path.join(output_dir, "master_stop_model_summary.json"), "w") as f:
            json.dump(master_summary_rows, f, indent=4, default=str)
        print(f"\n[StopStudy] Saved master summary to {output_dir}")
    else:
        print("\n[StopStudy] No results produced.")


if __name__ == "__main__":
    main()
