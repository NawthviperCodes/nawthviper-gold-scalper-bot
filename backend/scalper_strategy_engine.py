
# ======================================================
# === scalper_strategy_engine.py (Phase 1–4 Integrated) =
# ======================================================
#
# Research-backed live engine:
# - Uses stricter Phase 1C zone detector
# - Reversal-first decision engine
# - Candlesticks optional bonus only
# - Stop model handled inside decision engine (wick + 0.15 ATR)
# - Daily circuit breaker kept
# - Daily candle filter removed
#
import MetaTrader5 as mt5
import pandas as pd
import time as time_module
import json
import threading
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from zone_detector import detect_zones, detect_fast_zones
from news_filter_te import start_news_thread, check_upcoming_high_impact
from trade_executor import (
    place_order,
    trail_sl,
    close_partial_and_move_sl_to_be,
    close_positions_for_symbol,
    EXECUTION_LOCK,
)
from trade_decision_engine import run_trade_decision_engine, format_confidence_label, print_rejection_summary
from strategies import (
    ZoneReversalConfig,
    build_zone_reversal_snapshot,
    closed_bars_frame,
    normalize_pipeline_mode,
    quote_rejection_reason,
    run_zone_reversal_pipeline,
    wilder_atr,
)
from performance_tracker import print_performance_summary, log_trade
from trade_logger import log_pending_trade, update_trade_result
from telegram_notifier import send_telegram_message

PRINT_LOCK = Lock()


def load_config(path='config.json'):
    try:
        with open(path, 'r') as f:
            config = json.load(f)

        timeframe_map = {
            "TIMEFRAME_M1": mt5.TIMEFRAME_M1,
            "TIMEFRAME_M5": mt5.TIMEFRAME_M5,
            "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
            "TIMEFRAME_H1": mt5.TIMEFRAME_H1,
            "TIMEFRAME_H4": mt5.TIMEFRAME_H4,
        }

        config['StrategyParameters']['TIMEFRAME_ZONE'] = timeframe_map[config['StrategyParameters']['TIMEFRAME_ZONE']]
        config['StrategyParameters']['TIMEFRAME_ENTRY'] = timeframe_map[config['StrategyParameters']['TIMEFRAME_ENTRY']]
        config['StrategyParameters']['TIMEFRAME_CONFIRM'] = timeframe_map[config['StrategyParameters']['TIMEFRAME_CONFIRM']]
        tf_htf_str = config['StrategyParameters'].get('TIMEFRAME_HTF', "TIMEFRAME_H4")
        config['StrategyParameters']['TIMEFRAME_HTF'] = timeframe_map.get(tf_htf_str, mt5.TIMEFRAME_H4)
        return config
    except Exception as e:
        print(f"[CRITICAL] Config Load Failed: {e}")
        raise


DECISION_STATS = {
    "cycles": 0,
    "htf_neutral_blocks": 0,
    "no_zones_found": 0,
    "price_not_in_zone": 0,
    "news_blocked": 0,
    "cooldown_blocked": 0,
    "circuit_breaker_blocked": 0,
    "signals_generated": 0,
    "signals_executed": 0,
    "invalid_quotes": 0,
    "stale_quotes": 0,
    "future_quotes": 0,
    "pipeline_parity_checks": 0,
    "pipeline_parity_mismatches": 0,
}

ZONE_TOUCH_STATS = {
    "demand_touches": 0,
    "supply_touches": 0,
}

CONFIG = load_config('config.json')

SYMBOLS = CONFIG['BotSettings']['SYMBOLS']
MAGIC = CONFIG['BotSettings']['MAGIC']
DRY_RUN = CONFIG['BotSettings']['DRY_RUN']
MAX_QUOTE_AGE_SECONDS = float(CONFIG['BotSettings'].get('MAX_QUOTE_AGE_SECONDS', 120))
if MAX_QUOTE_AGE_SECONDS <= 0:
    raise ValueError("BotSettings.MAX_QUOTE_AGE_SECONDS must be positive")
STRATEGY_PIPELINE_MODE = normalize_pipeline_mode(
    CONFIG['BotSettings'].get('STRATEGY_PIPELINE_MODE', 'legacy')
)

TIMEFRAME_ZONE = CONFIG['StrategyParameters']['TIMEFRAME_ZONE']
TIMEFRAME_ENTRY = CONFIG['StrategyParameters']['TIMEFRAME_ENTRY']
TIMEFRAME_CONFIRM = CONFIG['StrategyParameters']['TIMEFRAME_CONFIRM']
TIMEFRAME_HTF = CONFIG['StrategyParameters']['TIMEFRAME_HTF']
MT5_TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M1: 'M1',
    mt5.TIMEFRAME_M5: 'M5',
    mt5.TIMEFRAME_M15: 'M15',
    mt5.TIMEFRAME_H1: 'H1',
    mt5.TIMEFRAME_H4: 'H4',
}
ZONE_LOOKBACK = CONFIG['StrategyParameters']['ZONE_LOOKBACK']
TP_RATIO = CONFIG['StrategyParameters']['TP_RATIO']
PARTIAL_CLOSE_PERCENT = CONFIG['StrategyParameters']['PARTIAL_CLOSE_PERCENT']
AUTO_CLOSE_ON_STRONG = CONFIG['StrategyParameters']['AUTO_CLOSE_ON_STRONG']

THRESHOLDS = dict(CONFIG['StrategyParameters'].get('Thresholds', {}))
THRESHOLDS['MAX_TOUCH_ALLOWED'] = CONFIG['StrategyParameters'].get('MAX_TOUCH_ALLOWED', 2)
CONFIDENCE_THRESHOLD = THRESHOLDS.get('MIN_CONFIDENCE_FOR_TRADE', 0.60)
MIN_CONF_FOR_TELEGRAM = THRESHOLDS.get('MIN_CONF_FOR_TELEGRAM', 0.60)

SYMBOL_SPECS = {}
tp1_hit_tickets = set()
_last_closure_time = {}
COOLDOWN_MINUTES = 15
MAX_DAILY_CONSECUTIVE_LOSSES = 3

_pending_signals = {}
_logged_deal_tickets = set()


def get_symbol_spec(symbol):
    if symbol not in SYMBOL_SPECS:
        info = mt5.symbol_info(symbol)
        if info:
            SYMBOL_SPECS[symbol] = info
    return SYMBOL_SPECS.get(symbol)


def safe_telegram(msg):
    try:
        threading.Thread(target=send_telegram_message, args=(msg,)).start()
    except Exception:
        pass


def send_info(msg):
    print(f"[ENGINE] {datetime.now().strftime('%H:%M:%S')} {msg}")


def _record_pipeline_parity(report):
    DECISION_STATS["pipeline_parity_checks"] += 1
    if report.matched:
        return
    DECISION_STATS["pipeline_parity_mismatches"] += 1
    count = DECISION_STATS["pipeline_parity_mismatches"]
    if count <= 5 or count % 100 == 0:
        send_info(
            f"[PIPELINE SHADOW] {report.symbol} mismatch #{count} at {report.as_of.isoformat()} "
            f"error={report.canonical_error or 'signal divergence'}"
        )


def get_data(symbol, timeframe, bars):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception:
        return pd.DataFrame()


def get_closed_data(symbol, timeframe, bars, as_of):
    """Return the requested number of fully closed MT5 candles as of a quote."""
    timeframe_name = MT5_TIMEFRAME_NAMES.get(timeframe)
    if timeframe_name is None:
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    raw = get_data(symbol, timeframe, bars + 2)
    if raw.empty:
        return raw
    return closed_bars_frame(
        raw,
        timeframe=timeframe_name,
        as_of=as_of,
        tail=bars,
    ).reset_index(drop=True)


def calculate_trend(df):
    if df.empty or len(df) < 51:
        return None
    sma50 = df['close'].rolling(50).mean().iloc[-1]
    last = df['close'].iloc[-1]
    return "uptrend" if last > sma50 else "downtrend"


def get_htf_bias(df):
    if df.empty or len(df) < 200:
        return "NEUTRAL"
    ema_fast = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    if ema_fast > ema_slow:
        return "UP"
    if ema_fast < ema_slow:
        return "DOWN"
    return "NEUTRAL"


def check_daily_circuit_breaker(symbol):
    try:
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)
        deals = mt5.history_deals_get(start_of_day, now, group=symbol)
        if deals is None or len(deals) == 0:
            return False

        consecutive_losses = 0
        for deal in reversed(deals):
            if deal.magic != MAGIC:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.profit < 0:
                consecutive_losses += 1
            elif deal.profit > 0:
                consecutive_losses = 0
                break

        return consecutive_losses >= MAX_DAILY_CONSECUTIVE_LOSSES
    except Exception as e:
        print(f"Circuit Breaker Error: {e}")
        return False


def check_and_log_closed_trades(symbol):
    try:
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)
        deals = mt5.history_deals_get(start_of_day, now, group=symbol)
        if not deals:
            return

        for deal in deals:
            if deal.magic != MAGIC:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.ticket in _logged_deal_tickets:
                continue

            position_ticket = deal.position_id
            signal = _pending_signals.get(position_ticket)
            if signal is None:
                signal = {"reason": "recovered-position", "strategy": "unknown"}

            total_profit = float(
                deal.profit + deal.commission + deal.swap + getattr(deal, "fee", 0.0)
            )
            result = "win" if total_profit > 0 else ("loss" if total_profit < 0 else "breakeven")
            close_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
            reason_map = {
                getattr(mt5, "DEAL_REASON_TP", -1): "TP",
                getattr(mt5, "DEAL_REASON_SL", -2): "SL",
                getattr(mt5, "DEAL_REASON_SO", -3): "kill-switch",
                getattr(mt5, "DEAL_REASON_CLIENT", -4): "manual",
                getattr(mt5, "DEAL_REASON_MOBILE", -5): "manual",
                getattr(mt5, "DEAL_REASON_WEB", -6): "manual",
                getattr(mt5, "DEAL_REASON_EXPERT", -7): "expert",
            }
            update_trade_result(
                position_id=position_ticket,
                exit_price=deal.price,
                close_time=close_time,
                close_reason=reason_map.get(deal.reason, "other"),
                profit=total_profit,
                result=result,
            )
            _last_closure_time[symbol] = close_time
            log_trade(signal, result)

            _logged_deal_tickets.add(deal.ticket)
            _pending_signals.pop(position_ticket, None)

            send_info(f"🧠 Logged: {symbol} {signal.get('reason')} [{signal.get('strategy')}] → {result.upper()} (${deal.profit:.2f})")
    except Exception as e:
        send_info(f"[LogClosed] Error for {symbol}: {e}")


def determine_lot_size(symbol, sl_price, entry_price, fixed_lot, strategy_mode):
    info = get_symbol_spec(symbol)
    if not info:
        return 0.01

    # Demo validation freezes sizing at exactly the coded 1% risk model.
    # Runtime fixed-lot inputs and performance multipliers are ignored.
    risk_percent = 1.0

    try:
        acc = mt5.account_info()
        if not acc:
            return info.volume_min

        equity = acc.equity
        risk_amt = equity * (risk_percent / 100.0)
        dist = abs(entry_price - sl_price)

        if dist == 0 or info.trade_tick_value == 0:
            return info.volume_min

        ticks = dist / info.trade_tick_size
        loss_per_lot = ticks * info.trade_tick_value
        if loss_per_lot <= 0:
            return info.volume_min

        lots = risk_amt / loss_per_lot
        step = info.volume_step
        lots = round(lots / step) * step
        return max(info.volume_min, min(info.volume_max, lots))
    except Exception:
        return info.volume_min


def check_for_partial_tp_live(symbol):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != MAGIC:
                continue
            if pos.ticket in tp1_hit_tickets:
                continue

            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            entry, sl = pos.price_open, pos.sl

            if (is_buy and sl >= entry) or ((not is_buy) and sl > 0 and sl <= entry):
                tp1_hit_tickets.add(pos.ticket)
                continue

            if sl == 0:
                continue
            risk = abs(entry - sl)
            target = (entry + risk) if is_buy else (entry - risk)

            tick = mt5.symbol_info_tick(symbol)
            curr = tick.bid if is_buy else tick.ask

            hit = (curr >= target) if is_buy else (curr <= target)
            if hit and PARTIAL_CLOSE_PERCENT > 0:
                send_info(f"💰 {symbol} Ticket {pos.ticket} hit 1R. Securing...")
                if not DRY_RUN and close_partial_and_move_sl_to_be(pos.ticket, PARTIAL_CLOSE_PERCENT):
                    tp1_hit_tickets.add(pos.ticket)
                    safe_telegram(f"💰 {symbol} 1R Secured!")
    except Exception:
        pass


def process_symbol_cycle(symbol, strategy_mode="zone_reversal_phase4", fixed_lot=None):
    try:
        DECISION_STATS["cycles"] += 1
        if not DRY_RUN:
            trail_sl(symbol, MAGIC)
            check_for_partial_tp_live(symbol)
        check_and_log_closed_trades(symbol)

        if check_upcoming_high_impact(symbol):
            DECISION_STATS["news_blocked"] += 1
            return

        if check_daily_circuit_breaker(symbol):
            DECISION_STATS["circuit_breaker_blocked"] += 1
            if DECISION_STATS["cycles"] % 100 == 0:
                send_info(f"⛔ {symbol} Circuit Breaker Active (3 Consecutive Losses). Sleeping.")
            return

        last_close = _last_closure_time.get(symbol)
        if last_close:
            elapsed = (datetime.now(timezone.utc) - last_close).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                DECISION_STATS["cooldown_blocked"] += 1
                return

        tick = mt5.symbol_info_tick(symbol)
        spec = get_symbol_spec(symbol)
        if tick is None or spec is None:
            return
        observed_at = datetime.now(timezone.utc)
        tick_time_msc = getattr(tick, 'time_msc', 0)
        tick_time = getattr(tick, 'time', 0)
        if tick_time_msc:
            as_of = datetime.fromtimestamp(tick_time_msc / 1000.0, tz=timezone.utc)
        elif tick_time:
            as_of = datetime.fromtimestamp(tick_time, tz=timezone.utc)
        else:
            as_of = None
        quote_reason = quote_rejection_reason(
            bid=getattr(tick, 'bid', None),
            ask=getattr(tick, 'ask', None),
            occurred_at=as_of,
            observed_at=observed_at,
            max_age_seconds=MAX_QUOTE_AGE_SECONDS,
        )
        if quote_reason is not None:
            DECISION_STATS[f"{quote_reason}_quotes"] += 1
            if DECISION_STATS["cycles"] % 100 == 0:
                send_info(f"{symbol} rejected {quote_reason} market quote")
            return

        h4_df = get_closed_data(symbol, TIMEFRAME_HTF, 250, as_of)
        htf_bias = get_htf_bias(h4_df)
        if htf_bias == "NEUTRAL":
            DECISION_STATS["htf_neutral_blocks"] += 1
            return

        h1_df = get_closed_data(symbol, TIMEFRAME_ZONE, ZONE_LOOKBACK, as_of)
        if len(h1_df) < 80:
            return

        h1_atr = wilder_atr(h1_df)

        m5_df = get_closed_data(symbol, TIMEFRAME_CONFIRM, 80, as_of)
        if len(m5_df) < 5:
            return

        demand_zones, supply_zones = detect_zones(h1_df)
        fast_demand, fast_supply = detect_fast_zones(h1_df)

        if not demand_zones and not supply_zones and not fast_demand and not fast_supply:
            DECISION_STATS["no_zones_found"] += 1
            return

        trend = calculate_trend(h1_df)

        active_trades_virtual = {}
        live_pos = mt5.positions_get(symbol=symbol)
        if live_pos:
            for p in live_pos:
                if p.magic == MAGIC:
                    active_trades_virtual[symbol] = {
                        'side': 'buy' if p.type == mt5.ORDER_TYPE_BUY else 'sell',
                        'ticket': p.ticket,
                        'volume': p.volume,
                        'entry_price': p.price_open,
                        'opened_at': datetime.fromtimestamp(p.time, tz=timezone.utc),
                        'stop_loss': p.sl,
                        'take_profit': p.tp,
                    }

        atr = wilder_atr(m5_df)
        m5_context = {'trend': calculate_trend(m5_df)}

        for z in demand_zones:
            if abs(tick.bid - z["price"]) <= max(atr, 50 * spec.point):
                ZONE_TOUCH_STATS["demand_touches"] += 1
                break
        else:
            for z in supply_zones:
                if abs(tick.bid - z["price"]) <= max(atr, 50 * spec.point):
                    ZONE_TOUCH_STATS["supply_touches"] += 1
                    break
            else:
                DECISION_STATS["price_not_in_zone"] += 1

        legacy_kwargs = dict(
            symbol=symbol,
            point=spec.point,
            current_price=tick.bid,
            trend=trend,
            demand_zones=demand_zones,
            supply_zones=supply_zones,
            fast_demand_zones=fast_demand,
            fast_supply_zones=fast_supply,
            m1_candles_for_crt=None,
            m5_candles_for_patterns=m5_df.iloc[-5:],
            active_trades=active_trades_virtual,
            zone_touch_counts={},
            SL_BUFFER=0,
            TP_RATIO=TP_RATIO,
            CHECK_RANGE=max(atr, 50 * spec.point),
            LOT_SIZE=spec.volume_min,
            MAGIC=MAGIC,
            strategy_mode=strategy_mode,
            atr=atr,
            htf_atr=h1_atr,
            m5_context=m5_context,
            htf_high=h1_df['high'].max(),
            htf_low=h1_df['low'].min(),
            last_closed_h1=h1_df.iloc[-1],
            htf_bias=htf_bias,
            thresholds=THRESHOLDS,
            strategy_active_override=True,
            spread=max(0.0, float(tick.ask) - float(tick.bid)),
        )
        snapshot = None
        adapter_config = None
        if STRATEGY_PIPELINE_MODE != 'legacy':
            snapshot = build_zone_reversal_snapshot(
                symbol=symbol,
                as_of=as_of,
                bid=tick.bid,
                ask=tick.ask,
                h4=h4_df,
                h1=h1_df,
                m5=m5_df,
                demand_zones=demand_zones,
                supply_zones=supply_zones,
                active_trades=active_trades_virtual,
            )
            adapter_config = ZoneReversalConfig.from_thresholds(
                strategy_id=strategy_mode,
                point=spec.point,
                tp_ratio=TP_RATIO,
                pattern_bars=5,
                strategy_active=None,
                thresholds=THRESHOLDS,
            )

        signals, _ = run_zone_reversal_pipeline(
            mode=STRATEGY_PIPELINE_MODE,
            legacy_kwargs=legacy_kwargs,
            snapshot=snapshot,
            config=adapter_config,
            decision_fn=run_trade_decision_engine,
            parity_sink=_record_pipeline_parity,
        )

        if not signals:
            return

        DECISION_STATS["signals_generated"] += len(signals)
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        sig = signals[0]
        confidence = float(sig.get('confidence', 0))
        side = sig['side']

        if confidence < CONFIDENCE_THRESHOLD:
            return

        if symbol in active_trades_virtual:
            existing = active_trades_virtual[symbol]
            if (
                not DRY_RUN
                and existing['side'] != side
                and AUTO_CLOSE_ON_STRONG
                and confidence > 0.85
            ):
                send_info(f"⚔️ {symbol} Auto-Close for Strong Signal")
                close_positions_for_symbol(symbol)
                time_module.sleep(1)

        if symbol not in active_trades_virtual:
            entry, sl, tp = sig['entry'], sig['sl'], sig['tp']
            current_strat = sig.get('strategy', strategy_mode)
            # Size against the executable side of the quote. Using the signal's
            # bid for buys would omit the spread and could risk more than 1%.
            sizing_entry = tick.ask if side == "buy" else tick.bid
            final_lot = determine_lot_size(
                symbol, sl, sizing_entry, fixed_lot, current_strat
            )

            if DRY_RUN:
                msg = (
                    f"📥 [DRY] SIGNAL: {symbol} | {side.upper()} {sig.get('reason')}\n"
                    f"Conf: {format_confidence_label(confidence)} | Lot: {final_lot}"
                )
                safe_telegram(msg)
            else:
                res = place_order(symbol, side, final_lot, MAGIC, comment="Nawthviper", sl=sl, tp=tp)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    DECISION_STATS["signals_executed"] += 1
                    live_positions = [
                        position for position in (mt5.positions_get(symbol=symbol) or [])
                        if position.magic == MAGIC
                    ]
                    position = max(live_positions, key=lambda item: item.time) if live_positions else None
                    position_id = position.ticket if position is not None else res.order
                    actual_fill = position.price_open if position is not None else res.price
                    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
                    risk_value = mt5.order_calc_profit(
                        order_type, symbol, float(final_lot), float(actual_fill), float(sl)
                    )
                    if risk_value is not None:
                        initial_risk_cash = abs(float(risk_value))
                    else:
                        initial_risk_cash = (
                            abs(float(actual_fill) - float(sl))
                            / float(spec.trade_tick_size)
                            * float(spec.trade_tick_value)
                            * float(final_lot)
                        )
                    open_time = (
                        datetime.fromtimestamp(position.time, tz=timezone.utc)
                        if position is not None
                        else datetime.now(timezone.utc)
                    )
                    logged_signal = dict(sig)
                    logged_signal.update({
                        "position_id": position_id,
                        "actual_fill_price": actual_fill,
                        "signal_price": entry,
                        "zone_quality": "A",
                        "htf_bias": htf_bias,
                    })
                    _pending_signals[position_id] = logged_signal
                    log_pending_trade(
                        trade_id=res.order,
                        position_id=position_id,
                        open_time=open_time,
                        symbol=symbol,
                        strategy=current_strat,
                        side=side,
                        reason=sig.get("reason", ""),
                        zone_price=sig.get("zone", {}).get("mid", ""),
                        zone_quality="A",
                        htf_bias=htf_bias,
                        confidence=confidence,
                        signal_price=entry,
                        actual_fill_price=actual_fill,
                        sl=sl,
                        tp=tp,
                        lot_size=final_lot,
                        initial_risk_cash=initial_risk_cash,
                    )
                    if confidence >= MIN_CONF_FOR_TELEGRAM:
                        msg = (
                            f"📥 SIGNAL: {symbol} | {side.upper()} {sig.get('reason')}\n"
                            f"Entry: {entry:.5f} | SL: {sl:.5f}\n"
                            f"Conf: {format_confidence_label(confidence)}"
                        )
                        safe_telegram(msg)
                    send_info(f"✅ {symbol} {side} Opened. Ticket: {res.order}")
                elif res:
                    send_info(f"❌ {symbol} Order Failed: {res.comment} (Code: {res.retcode})")

        if DECISION_STATS["cycles"] % 100 == 0:
            print_decision_summary()
            print_rejection_summary()
            print_performance_summary()

    except Exception as e:
        send_info(f"Cycle Error {symbol}: {e}")
        traceback.print_exc()


def print_decision_summary():
    with PRINT_LOCK:
        print("\n===== DECISION JOURNAL SUMMARY =====")
        for k, v in DECISION_STATS.items():
            print(f"{k}: {v}")
        print("Zone Touches:", ZONE_TOUCH_STATS)
        print("===================================\n")
