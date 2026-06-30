import csv
import os
import threading
from datetime import datetime, timezone


LOG_FILE = "demo_validation_trades.csv"
_LOG_LOCK = threading.Lock()

HEADER = [
    "trade_id", "position_id", "open_time_utc", "close_time_utc",
    "symbol", "strategy", "side", "entry_reason", "zone_price",
    "zone_quality", "htf_bias", "confidence", "signal_price",
    "actual_fill_price", "slippage_price", "sl", "tp", "lot_size",
    "initial_risk_price", "initial_risk_cash", "exit_price",
    "close_reason", "profit", "realized_r", "result",
]


def _timestamp(value=None):
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _adverse_slippage(side, signal_price, actual_fill_price):
    signal_price = float(signal_price)
    actual_fill_price = float(actual_fill_price)
    return (
        actual_fill_price - signal_price
        if str(side).lower() == "buy"
        else signal_price - actual_fill_price
    )


def log_pending_trade(
    *, trade_id, position_id, open_time, symbol, strategy, side, reason,
    zone_price, zone_quality, htf_bias, confidence, signal_price,
    actual_fill_price, sl, tp, lot_size, initial_risk_cash,
):
    """Persist the complete immutable entry snapshot for demo validation."""
    row = {
        "trade_id": str(trade_id),
        "position_id": str(position_id),
        "open_time_utc": _timestamp(open_time),
        "close_time_utc": "",
        "symbol": symbol,
        "strategy": strategy,
        "side": side,
        "entry_reason": reason,
        "zone_price": zone_price,
        "zone_quality": zone_quality,
        "htf_bias": htf_bias,
        "confidence": float(confidence),
        "signal_price": float(signal_price),
        "actual_fill_price": float(actual_fill_price),
        "slippage_price": _adverse_slippage(side, signal_price, actual_fill_price),
        "sl": float(sl),
        "tp": float(tp),
        "lot_size": float(lot_size),
        "initial_risk_price": abs(float(actual_fill_price) - float(sl)),
        "initial_risk_cash": abs(float(initial_risk_cash)),
        "exit_price": "",
        "close_reason": "",
        "profit": "",
        "realized_r": "",
        "result": "",
    }

    with _LOG_LOCK:
        exists = os.path.isfile(LOG_FILE)
        if exists:
            with open(LOG_FILE, "r", newline="", encoding="utf-8") as handle:
                if any(
                    existing.get("position_id") == row["position_id"]
                    for existing in csv.DictReader(handle)
                ):
                    return row["position_id"]
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    return row["position_id"]


def update_trade_result(
    *, position_id, exit_price, close_time, close_reason, profit, result=None,
):
    """Close an existing entry record and calculate realized R."""
    if not os.path.exists(LOG_FILE):
        return False

    updated = False
    rows = []
    with _LOG_LOCK:
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("position_id") == str(position_id)
                    and not row.get("close_time_utc")
                ):
                    risk_cash = abs(float(row.get("initial_risk_cash") or 0.0))
                    realized_r = float(profit) / risk_cash if risk_cash > 0 else ""
                    row.update({
                        "close_time_utc": _timestamp(close_time),
                        "exit_price": float(exit_price),
                        "close_reason": close_reason,
                        "profit": float(profit),
                        "realized_r": realized_r,
                        "result": result or (
                            "win" if float(profit) > 0
                            else "loss" if float(profit) < 0
                            else "breakeven"
                        ),
                    })
                    updated = True
                rows.append(row)

        if updated:
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADER)
                writer.writeheader()
                writer.writerows(rows)
    return updated
