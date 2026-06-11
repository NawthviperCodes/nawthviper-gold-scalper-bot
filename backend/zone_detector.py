# ==========================================
# zone_detector.py (Phase 1C Lifecycle Version)
# Stricter Supply/Demand Zone Engine
# ==========================================

import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numba import jit
from ta.volatility import AverageTrueRange


# ----------------------------------------------------------
# Public API
# ----------------------------------------------------------

def detect_zones(
    df: pd.DataFrame,
    lookback: int = 100,
    zone_size: int = 5,
    min_width_atr: float = 0.20,
    max_width_atr: float = 1.10,
    departure_atr_mult: float = 1.60,
    departure_bars: int = 4,
    merge_gap_atr: float = 0.20,
    max_age_bars: int = 180,
    max_touch_events: int = 2,
    full_touch_limit: int = 1,
    invalidation_close_penetration: float = 0.35,
):
    """
    Stricter lifecycle-aware supply/demand detector.

    Compatibility preserved:
      returns (demand_zones, supply_zones)
      each zone still includes: type, price, top, bottom, time, touches

    Phase 1C upgrades:
      - lookback actually applied
      - width ATR band tightened
      - departure threshold tightened
      - nearby same-type zones merged
      - zones expire after max_age_bars
      - touch counting grouped by events (not candles)
      - stronger invalidation rules
      - harsher quality scoring / freshness scoring
    """
    if df is None or len(df) < max(zone_size * 2 + 30, 60):
        return [], []

    work = df.tail(lookback).copy() if lookback and len(df) > lookback else df.copy()
    work = work.reset_index(drop=True)

    atr_values = _compute_atr(work)

    highs = work['high'].values.astype(np.float64)
    lows = work['low'].values.astype(np.float64)
    opens = work['open'].values.astype(np.float64)
    closes = work['close'].values.astype(np.float64)
    times = work['time'].astype('int64').values // 10**9

    demand_rows, supply_rows = _find_zones_numba(
        highs, lows, opens, closes, atr_values, times,
        zone_size=zone_size,
        departure_bars=departure_bars,
        departure_atr_mult=departure_atr_mult,
        min_width_atr=min_width_atr,
        max_width_atr=max_width_atr,
    )

    demand_zones = _rows_to_zones(demand_rows, 'demand')
    supply_zones = _rows_to_zones(supply_rows, 'supply')

    demand_zones = _merge_nearby_zones(demand_zones, atr_values, work, merge_gap_atr)
    supply_zones = _merge_nearby_zones(supply_zones, atr_values, work, merge_gap_atr)

    demand_zones = _apply_lifecycle(
        demand_zones, work, atr_values,
        max_age_bars=max_age_bars,
        max_touch_events=max_touch_events,
        full_touch_limit=full_touch_limit,
        invalidation_close_penetration=invalidation_close_penetration,
    )
    supply_zones = _apply_lifecycle(
        supply_zones, work, atr_values,
        max_age_bars=max_age_bars,
        max_touch_events=max_touch_events,
        full_touch_limit=full_touch_limit,
        invalidation_close_penetration=invalidation_close_penetration,
    )

    return demand_zones, supply_zones


def detect_fast_zones(df, point=0.0001, min_cluster=2, lookback=15):
    """Compatibility helper kept from prior versions."""
    fast_demand, fast_supply = [], []
    if df is None or len(df) < lookback:
        return [], []

    recent = df.tail(lookback)

    atr_points = (
        recent['close'].diff().abs().rolling(14).mean().iloc[-1]
        if len(recent) >= 14 else None
    )
    if pd.isna(atr_points) or atr_points is None:
        atr_points = point * 20

    proximity = max(atr_points * 1.5, point * 15)

    lows = recent['low'].values
    highs = recent['high'].values

    min_low = np.min(lows)
    max_high = np.max(highs)

    low_count = np.sum(np.abs(lows - min_low) <= proximity)
    high_count = np.sum(np.abs(highs - max_high) <= proximity)

    last_c = recent['close'].iloc[-1]
    last_l = recent['low'].iloc[-1]
    last_h = recent['high'].iloc[-1]

    if low_count >= min_cluster and (last_c - last_l) > atr_points * 0.5:
        fast_demand.append({
            'type': 'fast_demand',
            'price': float(min_low),
            'time': recent['time'].iloc[-1] if 'time' in recent.columns else recent.index[-1],
        })

    if high_count >= min_cluster and (last_h - last_c) > atr_points * 0.5:
        fast_supply.append({
            'type': 'fast_supply',
            'price': float(max_high),
            'time': recent['time'].iloc[-1] if 'time' in recent.columns else recent.index[-1],
        })

    return fast_demand, fast_supply


def update_zones(zones: List[dict], current_price: float) -> List[dict]:
    """Compatibility helper: lightweight runtime filter."""
    out = []
    for z in zones:
        if z['type'] == 'demand' and current_price >= z['bottom']:
            out.append(z)
        elif z['type'] == 'supply' and current_price <= z['top']:
            out.append(z)
        elif z['type'].startswith('fast_'):
            out.append(z)
    return out


def zones_to_research_df(zones: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(zones) if zones else pd.DataFrame()


def summarize_zone_set(zones: List[dict]) -> Dict[str, float]:
    if not zones:
        return {
            'count': 0,
            'avg_width_atr_ratio': 0.0,
            'avg_departure_strength_atr': 0.0,
            'avg_touches': 0.0,
            'fresh_avg': 0.0,
            'quality_counts': {},
        }
    df = pd.DataFrame(zones)
    return {
        'count': int(len(df)),
        'avg_width_atr_ratio': round(float(df['width_atr_ratio'].mean()), 4),
        'avg_departure_strength_atr': round(float(df['departure_strength_atr'].mean()), 4),
        'avg_touches': round(float(df['touches'].mean()), 2),
        'fresh_avg': round(float(df['freshness_score'].mean()), 4),
        'quality_counts': df['quality_bucket'].value_counts(dropna=False).to_dict(),
    }


# ----------------------------------------------------------
# Core detection (Numba)
# ----------------------------------------------------------

@jit(nopython=True)
def _find_zones_numba(
    highs, lows, opens, closes, atr, times,
    zone_size, departure_bars, departure_atr_mult,
    min_width_atr, max_width_atr,
):
    n = len(closes)
    demand_raw = []
    supply_raw = []

    start = max(zone_size, 20)
    end = n - zone_size - departure_bars

    for i in range(start, end):
        h = highs[i]
        l = lows[i]
        o = opens[i]
        c = closes[i]
        a = atr[i]
        t = times[i]

        if a <= 0:
            continue

        # pivot low
        pivot_low = True
        for j in range(1, zone_size + 1):
            if lows[i - j] <= l or lows[i + j] <= l:
                pivot_low = False
                break

        if pivot_low:
            body_top = o if c < o else c
            width = body_top - l
            width_atr_ratio = width / a if a > 0 else 0.0
            if width > 0 and width_atr_ratio >= min_width_atr and width_atr_ratio <= max_width_atr:
                max_dep = highs[i + 1]
                for k in range(2, departure_bars + 1):
                    if highs[i + k] > max_dep:
                        max_dep = highs[i + k]
                dep_abs = max_dep - l
                dep_atr = dep_abs / a if a > 0 else 0.0
                if dep_atr >= departure_atr_mult:
                    demand_raw.append((l, body_top, t, i, a, dep_abs, dep_atr, width, width_atr_ratio))

        # pivot high
        pivot_high = True
        for j in range(1, zone_size + 1):
            if highs[i - j] >= h or highs[i + j] >= h:
                pivot_high = False
                break

        if pivot_high:
            body_bottom = o if c > o else c
            width = h - body_bottom
            width_atr_ratio = width / a if a > 0 else 0.0
            if width > 0 and width_atr_ratio >= min_width_atr and width_atr_ratio <= max_width_atr:
                min_dep = lows[i + 1]
                for k in range(2, departure_bars + 1):
                    if lows[i + k] < min_dep:
                        min_dep = lows[i + k]
                dep_abs = h - min_dep
                dep_atr = dep_abs / a if a > 0 else 0.0
                if dep_atr >= departure_atr_mult:
                    supply_raw.append((h, body_bottom, t, i, a, dep_abs, dep_atr, width, width_atr_ratio))

    return demand_raw, supply_raw


# ----------------------------------------------------------
# Conversion and enrichment
# ----------------------------------------------------------

def _compute_atr(df: pd.DataFrame) -> np.ndarray:
    try:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        vals = atr.average_true_range().fillna(0.0).values.astype(np.float64)
        return vals
    except Exception:
        return np.zeros(len(df), dtype=np.float64)


def _rows_to_zones(rows, zone_type: str) -> List[dict]:
    zones = []
    for row in rows:
        extreme, body_edge, t, idx, atr_at_form, dep_abs, dep_atr, width, width_atr_ratio = row
        if zone_type == 'demand':
            bottom, top, price = float(extreme), float(body_edge), float(extreme)
        else:
            top, bottom, price = float(extreme), float(body_edge), float(extreme)

        zones.append({
            'type': zone_type,
            'price': price,
            'bottom': bottom,
            'top': top,
            'time': pd.to_datetime(int(t), unit='s'),
            'formation_idx': int(idx),
            'touches': 0,
            'tap_touches': 0,
            'deep_touches': 0,
            'full_touches': 0,
            'touch_events': [],
            'width': float(width),
            'width_atr_ratio': float(width_atr_ratio),
            'atr_at_formation': float(atr_at_form),
            'departure_strength_abs': float(dep_abs),
            'departure_strength_atr': float(dep_atr),
            'freshness_score': 1.0,
            'quality_bucket': 'C',
            'merged_from_count': 1,
            'age_bars': 0,
            'expired': False,
            'invalidated': False,
            'invalidation_reason': '',
            'last_touch_idx': None,
        })
    return zones


def _merge_nearby_zones(zones: List[dict], atr_values: np.ndarray, df: pd.DataFrame, merge_gap_atr: float) -> List[dict]:
    if not zones:
        return []

    zones = sorted(zones, key=lambda z: (z['formation_idx'], z['bottom'], z['top']))
    merged = []

    for z in zones:
        if not merged:
            merged.append(z)
            continue

        last = merged[-1]
        if last['type'] != z['type']:
            merged.append(z)
            continue

        atr_ref = max(last.get('atr_at_formation', 0.0), z.get('atr_at_formation', 0.0), 1e-9)
        gap = max(0.0, max(last['bottom'], z['bottom']) - min(last['top'], z['top']))
        mid_gap = abs(((last['top'] + last['bottom']) / 2.0) - ((z['top'] + z['bottom']) / 2.0))

        if gap <= atr_ref * merge_gap_atr or mid_gap <= atr_ref * merge_gap_atr:
            # merge into stricter composite zone
            if z['type'] == 'demand':
                last['bottom'] = min(last['bottom'], z['bottom'])
                last['top'] = min(last['top'], z['top']) if min(last['top'], z['top']) > last['bottom'] else max(last['top'], z['top'])
                last['price'] = last['bottom']
            else:
                last['top'] = max(last['top'], z['top'])
                last['bottom'] = max(last['bottom'], z['bottom']) if max(last['bottom'], z['bottom']) < last['top'] else min(last['bottom'], z['bottom'])
                last['price'] = last['top']

            last['formation_idx'] = min(last['formation_idx'], z['formation_idx'])
            last['time'] = min(last['time'], z['time'])
            last['width'] = float(max(last['top'] - last['bottom'], 1e-9))
            last['atr_at_formation'] = float(max(last['atr_at_formation'], z['atr_at_formation']))
            last['width_atr_ratio'] = float(last['width'] / max(last['atr_at_formation'], 1e-9))
            last['departure_strength_abs'] = float(max(last['departure_strength_abs'], z['departure_strength_abs']))
            last['departure_strength_atr'] = float(max(last['departure_strength_atr'], z['departure_strength_atr']))
            last['merged_from_count'] += 1
        else:
            merged.append(z)

    return merged


def _apply_lifecycle(
    zones: List[dict],
    df: pd.DataFrame,
    atr_values: np.ndarray,
    max_age_bars: int,
    max_touch_events: int,
    full_touch_limit: int,
    invalidation_close_penetration: float,
) -> List[dict]:
    if not zones:
        return []

    out = []
    for z in zones:
        final = _simulate_zone_lifecycle(
            z.copy(), df, atr_values,
            max_age_bars=max_age_bars,
            max_touch_events=max_touch_events,
            full_touch_limit=full_touch_limit,
            invalidation_close_penetration=invalidation_close_penetration,
        )
        if final is None:
            continue
        final['freshness_score'] = _freshness_score(final, max_age_bars=max_age_bars, max_touch_events=max_touch_events)
        final['quality_bucket'] = _quality_bucket(final)
        out.append(final)
    return out


def _simulate_zone_lifecycle(
    zone: dict,
    df: pd.DataFrame,
    atr_values: np.ndarray,
    max_age_bars: int,
    max_touch_events: int,
    full_touch_limit: int,
    invalidation_close_penetration: float,
):
    start = int(zone['formation_idx']) + 1
    n = len(df)
    width = max(float(zone['top'] - zone['bottom']), 1e-9)

    in_touch = False
    event_penetration = 0.0
    event_start = None

    for i in range(start, n):
        zone['age_bars'] = i - zone['formation_idx']
        candle = df.iloc[i]

        # expire stale untouched zones
        if zone['touches'] == 0 and zone['age_bars'] > max_age_bars:
            zone['expired'] = True
            zone['invalidation_reason'] = 'stale_untouched'
            return None

        overlap = candle['high'] >= zone['bottom'] and candle['low'] <= zone['top']

        if overlap:
            penetration_ratio = _penetration_ratio(candle, zone)
            if not in_touch:
                in_touch = True
                event_start = i
                event_penetration = penetration_ratio
            else:
                event_penetration = max(event_penetration, penetration_ratio)
        else:
            if in_touch:
                _close_touch_event(zone, event_start, i - 1, event_penetration)
                in_touch = False
                event_penetration = 0.0
                event_start = None

        # deep acceptance / close-through invalidation
        if _is_invalidated(candle, zone, invalidation_close_penetration):
            zone['invalidated'] = True
            zone['invalidation_reason'] = 'close_through'
            return None

        if zone['full_touches'] > full_touch_limit:
            zone['invalidated'] = True
            zone['invalidation_reason'] = 'too_many_full_touches'
            return None

        if zone['touches'] > max_touch_events:
            zone['expired'] = True
            zone['invalidation_reason'] = 'too_many_touch_events'
            return None

    if in_touch:
        _close_touch_event(zone, event_start, n - 1, event_penetration)

    zone['width'] = max(zone['top'] - zone['bottom'], 1e-9)
    return zone


def _penetration_ratio(candle: pd.Series, zone: dict) -> float:
    width = max(float(zone['top'] - zone['bottom']), 1e-9)
    if zone['type'] == 'demand':
        penetration = max(0.0, zone['top'] - float(candle['low']))
    else:
        penetration = max(0.0, float(candle['high']) - zone['bottom'])
    return penetration / width


def _close_touch_event(zone: dict, start_idx: int, end_idx: int, max_penetration_ratio: float):
    if start_idx is None:
        return
    touch_type = 'tap'
    if max_penetration_ratio > 0.75:
        touch_type = 'full'
    elif max_penetration_ratio > 0.20:
        touch_type = 'deep'

    zone['touch_events'].append({
        'start_idx': int(start_idx),
        'end_idx': int(end_idx),
        'max_penetration_ratio': float(max_penetration_ratio),
        'type': touch_type,
    })
    zone['touches'] += 1
    zone['last_touch_idx'] = int(end_idx)
    if touch_type == 'tap':
        zone['tap_touches'] += 1
    elif touch_type == 'deep':
        zone['deep_touches'] += 1
    else:
        zone['full_touches'] += 1


def _is_invalidated(candle: pd.Series, zone: dict, invalidation_close_penetration: float) -> bool:
    width = max(float(zone['top'] - zone['bottom']), 1e-9)
    if zone['type'] == 'demand':
        # invalid when candle closes below bottom OR closes too deeply inside lower part
        if float(candle['close']) < zone['bottom']:
            return True
        close_penetration = max(0.0, zone['top'] - float(candle['close'])) / width
        if float(candle['close']) < zone['top'] and close_penetration >= invalidation_close_penetration:
            return True
    else:
        if float(candle['close']) > zone['top']:
            return True
        close_penetration = max(0.0, float(candle['close']) - zone['bottom']) / width
        if float(candle['close']) > zone['bottom'] and close_penetration >= invalidation_close_penetration:
            return True
    return False


def _freshness_score(zone: dict, max_age_bars: int, max_touch_events: int) -> float:
    score = 1.0

    age_ratio = min(float(zone.get('age_bars', 0)) / max(max_age_bars, 1), 1.0)
    score -= age_ratio * 0.35

    touches = int(zone.get('touches', 0))
    score -= min(touches / max(max_touch_events, 1), 1.0) * 0.30

    score -= min(int(zone.get('deep_touches', 0)) * 0.10, 0.20)
    score -= min(int(zone.get('full_touches', 0)) * 0.18, 0.36)

    if int(zone.get('merged_from_count', 1)) > 1:
        score += 0.05

    return float(max(0.0, min(1.0, score)))


def _quality_bucket(zone: dict) -> str:
    dep = float(zone.get('departure_strength_atr', 0.0))
    width_ratio = float(zone.get('width_atr_ratio', 999.0))
    fresh = float(zone.get('freshness_score', 0.0))
    touches = int(zone.get('touches', 0))
    full_touches = int(zone.get('full_touches', 0))

    score = 0.0

    # Departure needs to be genuinely strong
    if dep >= 3.0:
        score += 0.45
    elif dep >= 2.2:
        score += 0.30
    elif dep >= 1.6:
        score += 0.15

    # Ideal width band is moderate, not tiny and not broad
    if 0.35 <= width_ratio <= 0.90:
        score += 0.30
    elif 0.25 <= width_ratio <= 1.10:
        score += 0.18
    else:
        score += 0.00

    # Freshness matters a lot now
    score += fresh * 0.25

    # Penalties
    if touches >= 2:
        score -= 0.10
    if full_touches >= 1:
        score -= 0.20

    if score >= 0.82:
        return 'A'
    if score >= 0.52:
        return 'B'
    return 'C'
