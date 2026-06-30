# trade_executor.py
import MetaTrader5 as mt5
import time
import threading

# GLOBAL LOCK FOR ORDER SENDING
EXECUTION_LOCK = threading.Lock()

# ==========================================
# === SAFE CONSTANTS (Prevents Crashes) ===
# ==========================================
# Some MT5 python versions miss these constants, so we define them safely.
SYMBOL_FILLING_FOK = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
SYMBOL_FILLING_IOC = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)

def get_position_by_ticket(ticket):
    try:
        res = mt5.positions_get(ticket=ticket)
        if res: return res[0]
    except: pass
    return None

def get_filling_mode(symbol):
    """
    Automatically determines the correct filling mode for the symbol/broker.
    Many brokers reject IOC, so we fallback to FOK or RETURN.
    """
    info = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_FOK  # Default fallback
        
    filling = info.filling_mode
    
    # Priority: FOK > IOC > RETURN
    # We use the safe global constants defined above
    if filling & SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling & SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def close_partial_and_move_sl_to_be(ticket, partial_percent=0.5):
    with EXECUTION_LOCK:
        pos = get_position_by_ticket(ticket)
        if not pos: return False
        
        symbol = pos.symbol
        vol_close = round(pos.volume * partial_percent, 2)
        
        # Determine filling mode
        filling_type = get_filling_mode(symbol)

        req_close = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": vol_close,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
            "magic": pos.magic,
            "comment": "Partial TP",
            "type_filling": filling_type,
        }
        mt5.order_send(req_close)
        
        time.sleep(0.2)
        pos_new = get_position_by_ticket(ticket)
        if not pos_new: return True 
        
        req_sl = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": pos.price_open,
            "tp": pos_new.tp
        }
        res = mt5.order_send(req_sl)
        return res.retcode == mt5.TRADE_RETCODE_DONE

def place_order(symbol, side, lot, magic, comment="", sl=None, tp=None):
    with EXECUTION_LOCK:
        # initialize check
        if not mt5.initialize(): 
            print(f"[Exec] MT5 Init Failed")
            return None

        # One account-wide position at most. This check shares the order-send
        # lock, so simultaneous symbol signals cannot both pass before either
        # order is sent.
        positions = mt5.positions_get()
        if positions is None:
            print(f"[Exec] Position query failed; blocking {symbol} order")
            return None
        if len(positions) > 0:
            print(f"[Exec] Global one-position limit blocked {symbol} order")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick: 
            print(f"[Exec] No Tick Data for {symbol}")
            return None
        
        type_op = mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == 'buy' else tick.bid
        
        # Auto-detect correct filling mode
        filling_type = get_filling_mode(symbol)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": type_op,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type, # ✅ FIXED: Safe Dynamic Filling Mode
        }
        
        if sl: request["sl"] = float(sl)
        if tp: request["tp"] = float(tp)
        
        result = mt5.order_send(request)
        
        if result is None:
            print(f"[Exec] Order Send returned None for {symbol}")
            return None
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Print specific error to console for debugging
            print(f"[Exec] ERROR placing {symbol}: {result.comment} (RetCode: {result.retcode})")
            return result # Return result to let engine handle the error
            
        return result

def modify_position_sltp(ticket, sl, tp):
    with EXECUTION_LOCK:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp)
        }
        res = mt5.order_send(request)
        if res is None: return False
        return res.retcode == mt5.TRADE_RETCODE_DONE

def trail_sl(symbol, magic, trail_pips=150):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return

    info = mt5.symbol_info(symbol)
    if not info: return
    point = info.point
    dist = trail_pips * point
    
    for pos in positions:
        if pos.magic != magic: continue
        
        new_sl = None
        if pos.type == mt5.ORDER_TYPE_BUY:
            current_price = mt5.symbol_info_tick(symbol).bid
            proposed = current_price - dist
            if proposed > pos.price_open and proposed > pos.sl:
                new_sl = proposed
        else:
            current_price = mt5.symbol_info_tick(symbol).ask
            proposed = current_price + dist
            if proposed < pos.price_open and (pos.sl == 0 or proposed < pos.sl):
                new_sl = proposed
                
        if new_sl:
            with EXECUTION_LOCK:
                req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp}
                mt5.order_send(req)

def close_positions_for_symbol(symbol):
    with EXECUTION_LOCK:
        positions = mt5.positions_get(symbol=symbol)
        if not positions: return

        filling_type = get_filling_mode(symbol)

        for pos in positions:
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            if not tick: continue
            price = tick.bid if is_buy else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Auto Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }
            mt5.order_send(request)
            print(f"[Exec] Closed {pos.ticket} for {symbol}")
