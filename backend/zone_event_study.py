import os
import json
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from zone_detector import detect_zones


DEFAULT_CONFIG = {
    "data_dir": "./backtest_data",
    "output_dir": "./zone_research_results",
    "symbols": ["XAUUSDz", "EURUSDz", "USDJPYz"],
    "zone_timeframe": "H1",
    "detector_history_bars": 600,
    "warmup_bars": 250,
    "reaction_horizon_bars": 12,
    "min_zone_gap_points": 1e-9,
    "lookback": 100,
    "zone_size": 5,
}


def load_csv(path: str) -> pd.DataFrame:
    """
    Load MT5-exported CSV and standardise columns.

    Handles MT5's tab-separated format:
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    2021.01.03 23:00:00 1909.401 ...

    Also handles comma-separated and already-clean formats.
    """
    # Auto-detect separator by reading first line raw
    with open(path, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(path, sep=sep)

    # Strip whitespace and angle brackets from ALL column names, lowercase
    df.columns = [c.strip().strip('<>').strip().lower() for c in df.columns]

    # Rename MT5 column variants to standard names
    rename_map = {
        'tickvol': 'tick_volume',
        'tick_vol': 'tick_volume',
        'vol': 'real_volume',
        'volume': 'real_volume',
    }
    df.rename(columns=rename_map, inplace=True)

    # Build unified datetime column
    if 'date' in df.columns:
        date_str = df['date'].astype(str).str.replace('.', '-', regex=False)

        if 'time' in df.columns:
            time_str = df['time'].astype(str)
            df['time'] = pd.to_datetime(
                date_str + ' ' + time_str,
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )
        else:
            df['time'] = pd.to_datetime(
                date_str,
                format='%Y-%m-%d',
                errors='coerce'
            )

    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(
            df['time'].astype(str).str.replace('.', '-', regex=False),
            errors='coerce'
        )
    else:
        raise ValueError(
            f"Cannot find date/time column in {path}. Columns found: {list(df.columns)}"
        )

    # Validate OHLC
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Columns: {list(df.columns)}")

    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    return df[['time', 'open', 'high', 'low', 'close']].copy()


def zone_key(zone: dict) -> Tuple:
    return (
        zone.get('type'),
        round(float(zone.get('bottom', 0.0)), 8),
        round(float(zone.get('top', 0.0)), 8),
        str(pd.to_datetime(zone.get('time'))),
    )


def overlaps_zone(candle: pd.Series, zone: dict) -> bool:
    return candle['high'] >= zone['bottom'] and candle['low'] <= zone['top']


def is_invalidated(candle: pd.Series, zone: dict) -> bool:
    if zone['type'] == 'demand':
        return candle['close'] < zone['bottom']
    return candle['close'] > zone['top']


def classify_first_touch_depth(candle: pd.Series, zone: dict) -> str:
    width = max(zone['top'] - zone['bottom'], 1e-9)
    if zone['type'] == 'demand':
        penetration = max(0.0, zone['top'] - candle['low'])
    else:
        penetration = max(0.0, candle['high'] - zone['bottom'])

    ratio = penetration / width
    if ratio <= 0.20:
        return 'tap'
    if ratio <= 0.75:
        return 'deep'
    return 'full'


def reaction_target(zone: dict) -> float:
    width_value = zone.get('width')
    if width_value is None or pd.isna(width_value):
        width_value = float(zone['top']) - float(zone['bottom'])
    width = max(float(width_value), 1e-9)
    atr = float(zone.get('atr_at_formation', 0.0) or 0.0)
    return max(width, atr * 0.50, 1e-9)


def bar_close_timestamp(df: pd.DataFrame, idx: int) -> Optional[pd.Timestamp]:
    """Return when bar ``idx`` became fully observable.

    MT5 research files timestamp bars at their open.  The next bar's open is
    therefore the previous bar's close.  For the final bar, infer the regular
    interval only for audit metadata; it is never used to expose future OHLC.
    """
    if idx < 0 or idx >= len(df):
        return None
    if idx + 1 < len(df):
        return pd.Timestamp(df.iloc[idx + 1]['time'])
    if len(df) >= 2:
        deltas = pd.Series(df['time']).diff().dropna()
        if not deltas.empty:
            return pd.Timestamp(df.iloc[idx]['time']) + deltas.median()
    return None


def classify_reaction(df: pd.DataFrame, touch_idx: int, zone: dict, horizon: int) -> Tuple[str, Optional[int]]:
    """Classify only price action observable after the touch bar closes.

    A touch candle's completed high/low cannot be counted as post-touch
    reaction because their intrabar order relative to the touch is unknown.
    The only touch-bar outcome used is close-through invalidation, which is
    knowable when the touch bar closes.  Otherwise evaluation begins on the
    next bar and spans exactly ``horizon`` post-touch bars.
    """
    if touch_idx < 0 or touch_idx >= len(df):
        raise IndexError("touch_idx is outside the price dataframe")

    horizon = max(int(horizon), 1)
    target = reaction_target(zone)
    touch_candle = df.iloc[touch_idx]

    # A close through the zone on the touch bar is a confirmed breakout at
    # that bar's close, not a tradable post-touch reversal.
    if is_invalidated(touch_candle, zone):
        return 'breakout', touch_idx

    start_idx = touch_idx + 1
    if start_idx >= len(df):
        return 'right_censored', None
    end_idx = min(len(df) - 1, start_idx + horizon - 1)

    if zone['type'] == 'demand':
        reversal_level = max(touch_candle['close'], zone['top']) + target
        for j in range(start_idx, end_idx + 1):
            c = df.iloc[j]
            if c['close'] < zone['bottom']:
                return 'breakout', j
            if c['high'] >= reversal_level:
                return ('immediate_reversal' if j == start_idx else 'consolidation_then_reversal'), j
    else:
        reversal_level = min(touch_candle['close'], zone['bottom']) - target
        for j in range(start_idx, end_idx + 1):
            c = df.iloc[j]
            if c['close'] > zone['top']:
                return 'breakout', j
            if c['low'] <= reversal_level:
                return ('immediate_reversal' if j == start_idx else 'consolidation_then_reversal'), j

    if (end_idx - start_idx + 1) < horizon:
        return 'right_censored', None
    return 'no_clear_reaction', None


def summarize_symbol(zones_df: pd.DataFrame) -> dict:
    if zones_df.empty:
        return {
            'total_zones': 0,
            'demand_zones': 0,
            'supply_zones': 0,
            'touched_pct': 0.0,
            'invalidated_pct': 0.0,
            'avg_width_atr_ratio': 0.0,
            'avg_departure_strength_atr': 0.0,
            'avg_touches': 0.0,
            'reaction_counts': {},
            'quality_counts': {},
            'touch_depth_counts': {},
        }

    reaction_counts = zones_df['reaction_outcome'].value_counts(dropna=False).to_dict()
    quality_counts = zones_df['quality_bucket'].value_counts(dropna=False).to_dict() if 'quality_bucket' in zones_df.columns else {}
    touch_depth_counts = zones_df['first_touch_depth'].value_counts(dropna=False).to_dict()

    return {
        'total_zones': int(len(zones_df)),
        'demand_zones': int((zones_df['zone_type'] == 'demand').sum()),
        'supply_zones': int((zones_df['zone_type'] == 'supply').sum()),
        'touched_pct': round(float(zones_df['touched'].mean()) * 100, 2),
        'invalidated_pct': round(float(zones_df['invalidated'].mean()) * 100, 2),
        'avg_width_atr_ratio': round(float(zones_df['width_atr_ratio'].replace([np.inf, -np.inf], np.nan).dropna().mean()), 4) if 'width_atr_ratio' in zones_df else 0.0,
        'avg_departure_strength_atr': round(float(zones_df['departure_strength_atr'].replace([np.inf, -np.inf], np.nan).dropna().mean()), 4) if 'departure_strength_atr' in zones_df else 0.0,
        'avg_touches': round(float(zones_df['touches'].mean()), 2),
        'reaction_counts': reaction_counts,
        'quality_counts': quality_counts,
        'touch_depth_counts': touch_depth_counts,
    }


def run_symbol_study(symbol: str, df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    history_bars = int(cfg['detector_history_bars'])
    warmup = int(cfg['warmup_bars'])
    horizon = int(cfg['reaction_horizon_bars'])
    lookback = int(cfg['lookback'])
    zone_size = int(cfg['zone_size'])

    active: Dict[Tuple, dict] = {}
    completed: List[dict] = []

    # MT5 bar timestamps are unique in the research files.  Resolve detector
    # metadata by timestamp because detector indices are relative to its
    # rolling lookback window, not to the full symbol dataframe.
    index_by_time = {pd.Timestamp(t): int(idx) for idx, t in enumerate(df['time'])}

    for i in range(warmup, len(df)):
        hist_start = max(0, i - history_bars + 1)
        hist = df.iloc[hist_start:i+1].copy()
        if len(hist) < max(warmup, zone_size * 2 + 20):
            continue

        demand_zones, supply_zones = detect_zones(hist, lookback=lookback, zone_size=zone_size)
        current_zones = demand_zones + supply_zones

        for z in current_zones:
            created_at = pd.Timestamp(z.get('created_at', z.get('time')))
            activated_at = pd.Timestamp(z.get('activated_at', z.get('detected_at')))
            created_idx = index_by_time.get(created_at)
            activation_idx = index_by_time.get(activated_at)

            if created_idx is None or activation_idx is None:
                continue

            # Admit a candidate exactly when it becomes knowable.  In
            # particular, do not load zones that activated before ``warmup``
            # and merely survived until the study started; that would select
            # on future survival and undercount failed zones.
            if activation_idx != i:
                continue

            k = zone_key(z)
            if k in active:
                continue

            atr_form = z.get('atr_at_formation', np.nan)
            record = {
                'symbol': symbol,
                'zone_type': z['type'],
                'zone_time': str(created_at),
                'created_at': str(created_at),
                'detection_time': str(df.iloc[i]['time']),
                'activated_at': str(activated_at),
                'formation_idx': created_idx,
                'detection_idx': i,
                'activation_idx': activation_idx,
                'zone_bottom': float(z['bottom']),
                'zone_top': float(z['top']),
                'zone_mid': float(z['price']),
                'width': float(z.get('width', z['top'] - z['bottom'])),
                'width_atr_ratio': float(z.get('width_atr_ratio', np.nan)),
                'departure_strength_abs': float(z.get('departure_strength_abs', np.nan)),
                'departure_strength_atr': float(z.get('departure_strength_atr', np.nan)),
                'freshness_score': float(z.get('freshness_score', np.nan)),
                'quality_bucket': z.get('quality_bucket', 'unknown'),
                'touches': int(z.get('touches', 0)),
                'tap_touches': int(z.get('tap_touches', 0)),
                'deep_touches': int(z.get('deep_touches', 0)),
                'full_touches': int(z.get('full_touches', 0)),
                'merged_from_count': int(z.get('merged_from_count', 1)),
                'atr_at_formation': float(atr_form) if atr_form is not None else np.nan,
                'touched': False,
                'first_touch_time': '',
                'first_touch_idx': None,
                'touch_observed_at': '',
                'first_touch_depth': '',
                'invalidated': False,
                'invalidated_time': '',
                'reaction_outcome': 'untouched',
                'reaction_time': '',
                'reaction_idx': None,
                'reaction_bar_time': '',
                'reaction_evaluation_start_idx': None,
                'reaction_evaluation_start_time': '',
            }
            active[k] = record

        candle = df.iloc[i]
        to_remove = []
        for k, rec in active.items():
            if i <= rec['activation_idx']:
                continue

            zone = {
                'type': rec['zone_type'],
                'bottom': rec['zone_bottom'],
                'top': rec['zone_top'],
                'width': rec['width'],
                'atr_at_formation': rec['atr_at_formation'],
            }

            if not rec['touched'] and overlaps_zone(candle, zone):
                rec['touched'] = True
                rec['first_touch_time'] = str(candle['time'])
                rec['first_touch_idx'] = i
                touch_observed_at = bar_close_timestamp(df, i)
                rec['touch_observed_at'] = str(touch_observed_at) if touch_observed_at is not None else ''
                rec['first_touch_depth'] = classify_first_touch_depth(candle, zone)

                evaluation_start_idx = i + 1
                rec['reaction_evaluation_start_idx'] = evaluation_start_idx
                if evaluation_start_idx < len(df):
                    rec['reaction_evaluation_start_time'] = str(df.iloc[evaluation_start_idx]['time'])

                outcome, outcome_idx = classify_reaction(df, i, zone, horizon)
                rec['reaction_outcome'] = outcome
                if outcome_idx is not None:
                    rec['reaction_idx'] = outcome_idx
                    rec['reaction_bar_time'] = str(df.iloc[outcome_idx]['time'])
                    observed_at = bar_close_timestamp(df, outcome_idx)
                    rec['reaction_time'] = str(observed_at) if observed_at is not None else ''
                else:
                    evaluation_end_idx = min(len(df) - 1, i + horizon)
                    observed_at = bar_close_timestamp(df, evaluation_end_idx)
                    rec['reaction_time'] = str(observed_at) if observed_at is not None else ''

                if outcome in ('immediate_reversal', 'consolidation_then_reversal', 'breakout', 'no_clear_reaction', 'right_censored'):
                    completed.append(rec.copy())
                    to_remove.append(k)
                    continue

            if not rec['invalidated'] and is_invalidated(candle, zone):
                rec['invalidated'] = True
                rec['invalidated_time'] = str(candle['time'])
                if not rec['touched']:
                    rec['reaction_outcome'] = 'invalidated_before_touch'
                elif rec['reaction_outcome'] == 'untouched':
                    rec['reaction_outcome'] = 'breakout'
                completed.append(rec.copy())
                to_remove.append(k)

        for k in to_remove:
            active.pop(k, None)

    for rec in active.values():
        completed.append(rec.copy())

    zones_df = pd.DataFrame(completed)
    if zones_df.empty:
        return zones_df, summarize_symbol(zones_df)

    zones_df['zone_time'] = pd.to_datetime(zones_df['zone_time'])
    for col in [
        'created_at', 'detection_time', 'activated_at', 'touch_observed_at',
        'reaction_bar_time', 'reaction_evaluation_start_time',
    ]:
        if col in zones_df.columns:
            zones_df[col] = pd.to_datetime(zones_df[col], errors='coerce')
    if 'first_touch_time' in zones_df.columns:
        zones_df['first_touch_time'] = pd.to_datetime(zones_df['first_touch_time'], errors='coerce')
    if 'invalidated_time' in zones_df.columns:
        zones_df['invalidated_time'] = pd.to_datetime(zones_df['invalidated_time'], errors='coerce')
    if 'reaction_time' in zones_df.columns:
        zones_df['reaction_time'] = pd.to_datetime(zones_df['reaction_time'], errors='coerce')

    summary = summarize_symbol(zones_df)
    return zones_df, summary


def load_runtime_config(args) -> dict:
    cfg = DEFAULT_CONFIG.copy()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cfg['symbols'] = raw.get('BotSettings', {}).get('SYMBOLS', cfg['symbols'])

    if args.data_dir:
        cfg['data_dir'] = args.data_dir
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.symbols:
        cfg['symbols'] = [s.strip() for s in args.symbols.split(',') if s.strip()]

    return cfg


def main():
    parser = argparse.ArgumentParser(description='Detector-only zone event study for Phase 1 research.')
    parser.add_argument('--config', default='config.json', help='Optional config file to read symbols from')
    parser.add_argument('--data-dir', default=None, help='Folder containing historical CSVs (same folder you use for backtests)')
    parser.add_argument('--output-dir', default=None, help='Folder to save zone research results')
    parser.add_argument('--symbols', default=None, help='Comma-separated symbols override')
    args = parser.parse_args()

    cfg = load_runtime_config(args)
    data_dir = cfg['data_dir']
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ZoneStudy] Data dir:   {data_dir}")
    print(f"[ZoneStudy] Output dir: {output_dir}")
    print(f"[ZoneStudy] Symbols:    {cfg['symbols']}")

    master_rows = []
    for symbol in cfg['symbols']:
        print(f"\n{'─'*55}\n  Zone Study: {symbol}\n{'─'*55}")
        path = os.path.join(data_dir, f"{symbol}_{cfg['zone_timeframe']}.csv")
        if not os.path.exists(path):
            print(f"  [!] Missing file: {path}")
            continue

        df = load_csv(path)
        zones_df, summary = run_symbol_study(symbol, df, cfg)

        zones_csv = os.path.join(output_dir, f"{symbol}_zones.csv")
        summary_json = os.path.join(output_dir, f"{symbol}_zone_summary.json")
        zones_df.to_csv(zones_csv, index=False)
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=4, default=str)

        print(f"  Zones found:        {summary['total_zones']}")
        print(f"  Touched %:          {summary['touched_pct']}%")
        print(f"  Invalidated %:      {summary['invalidated_pct']}%")
        print(f"  Avg width/ATR:      {summary['avg_width_atr_ratio']}")
        print(f"  Avg departure ATR:  {summary['avg_departure_strength_atr']}")
        print(f"  Avg touches:        {summary['avg_touches']}")
        print(f"  Reactions:          {summary['reaction_counts']}")
        print(f"  Quality buckets:    {summary['quality_counts']}")
        print(f"  Touch depths:       {summary['touch_depth_counts']}")

        row = {'symbol': symbol, **summary}
        master_rows.append(row)

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_df.to_csv(os.path.join(output_dir, 'master_zone_summary.csv'), index=False)
        with open(os.path.join(output_dir, 'master_zone_summary.json'), 'w') as f:
            json.dump(master_rows, f, indent=4, default=str)
        print(f"\n[ZoneStudy] Saved master summary to {output_dir}")
    else:
        print("\n[ZoneStudy] No symbol results were produced.")


if __name__ == '__main__':
    main()
