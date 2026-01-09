import telegram
import configparser
from telegram import Update
from telegram.ext import CallbackContext, MessageHandler, Filters, Updater
import logging
import time
import re
import os
import MetaTrader5 as mt5
import math
import platform

from ai_signal_utils import gemini_parse_trade_intent_sync

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read the token and your user ID from the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

my_username = config.get('Telegram', 'my_username')
my_user_id = int(config.get('Telegram', 'my_user_id'))
token = config.get('TradingBot', 'token')
lot_size = float(config.get('Settings', 'lot_size'))

mt5_login = config.getint('MetaTrader', 'login', fallback=None)
mt5_password = config.get('MetaTrader', 'password', fallback=None)
mt5_server = config.get('MetaTrader', 'server', fallback=None)
mt5_terminal_path = config.get('MetaTrader', 'terminal_path', fallback=None)


def _safe_notify(context: CallbackContext, message: str) -> None:
    try:
        context.bot.send_message(my_user_id, message)
    except telegram.error.Unauthorized:
        logging.warning(
            "Bot can't DM user_id=%s. Open the bot chat and press Start (send /start), "
            "or verify Telegram.my_user_id in config.ini.",
            my_user_id,
        )
    except Exception:
        logging.exception("Failed sending Telegram notification to user_id=%s", my_user_id)


def _find_mt5_terminal_path() -> str | None:
    """Best-effort lookup of terminal64.exe on Windows.

    MetaTrader5 Python package requires a local MetaTrader 5 *x64 terminal*.
    """
    if mt5_terminal_path and os.path.exists(mt5_terminal_path):
        return mt5_terminal_path

    if os.name != 'nt':
        return None

    candidates = []
    program_files = os.environ.get('ProgramFiles')
    program_files_x86 = os.environ.get('ProgramFiles(x86)')

    for base in [program_files, program_files_x86]:
        if not base:
            continue
        candidates.extend([
            os.path.join(base, 'MetaTrader 5', 'terminal64.exe'),
            os.path.join(base, 'MetaTrader 5', 'terminal.exe'),
        ])

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def _print_mt5_not_found_help(err):
    logging.error("MetaTrader5 initialize() failed: %s", err)
    print("\nMetaTrader 5 terminal not found (IPC initialize failed).")
    print("Fix checklist:")
    print("  1) Install MetaTrader 5 x64 terminal on this machine.")
    print("  2) Launch MT5 once and log in to your trading account.")
    print("  3) Ensure you're running 64-bit Python (MetaTrader5 package is x64-only).")
    print("     Quick check: py -c \"import platform; print(platform.architecture())\"")
    print("  4) (Optional) Set [MetaTrader] terminal_path in config.ini to terminal64.exe")
    print("     Example: terminal_path = C:/Program Files/MetaTrader 5/terminal64.exe\n")

def initialize_bot():
    if os.name == 'nt' and platform.architecture()[0] != '64bit':
        raise SystemExit(
            "This bot requires 64-bit Python on Windows (MetaTrader5 x64). "
            "Install 64-bit Python and re-install requirements."
        )

    terminal_path = _find_mt5_terminal_path()

    # mt5.initialize requires a local terminal; login/password/server are optional but recommended.
    init_kwargs = {}
    if terminal_path:
        init_kwargs['path'] = terminal_path
    if mt5_login is not None and mt5_password and mt5_server:
        init_kwargs.update({'login': mt5_login, 'password': mt5_password, 'server': mt5_server})

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        # (-10003, 'IPC initialize failed, MetaTrader 5 x64 not found')
        if isinstance(err, tuple) and len(err) >= 1 and err[0] == -10003:
            _print_mt5_not_found_help(err)
        else:
            logging.error("initialize() failed, error code = %s", err)
            if terminal_path:
                print(f"Tried terminal path: {terminal_path}")
            else:
                print("No MT5 terminal path detected. Set [MetaTrader] terminal_path in config.ini")
        raise SystemExit(1)

    info = mt5.account_info()
    print(info)

def handle_message(update: Update, context: CallbackContext) -> None:
    text = None
    if update.message:
        text = update.message.text
    elif update.channel_post:
        text = update.channel_post.text
    if not text:
        return

    # Handle explicit management messages even if symbol gating changes.
    if re.search(r'close half lots', text, re.IGNORECASE):
        print(f"Received close half lots message: {text}")
        _safe_notify(context, f"Received a close half lots message: \n\n{text}")

        sl_match = re.search(r'MOVE SL to ([\d.]+)', text)
        if sl_match:
            new_sl = float(sl_match.group(1))
            symbol = extract_symbol(text)
            if symbol:
                close_half_and_update_sl(symbol, new_sl)
                _safe_notify(context, f"Closed half position and updated SL for {symbol} to {new_sl}")
                return
            else:
                print("Could not extract symbol from the message")
        else:
            print("Could not extract new SL value from the message")
        return

    # Only attempt trade parsing if message likely contains a symbol and BUY/SELL.
    # Accept common variants like BuyAbove/SellBelow.
    if re.search(r'\b(BUY|SELL)(?:\s*(?:ABOVE|BELOW))?\b|\b(BUYABOVE|SELLBELOW)\b', text, re.IGNORECASE) is None:
        return
    if re.search(r'\b([A-Z][A-Z0-9_.]{2,15})\b', text, re.IGNORECASE) is None:
        return

    print(f"Received matching message: {text}")
    _safe_notify(context, f"Received a new matching message: \n\n{text}")

    allowed_symbols = mt5_symbol_candidates_for_message(text)
    ai_intent = gemini_parse_trade_intent_sync(
        text,
        default_lot=float(lot_size),
        allowed_symbols=allowed_symbols,
    )

    # Debug: print the normalized AI response
    try:
        import json as _json
        print("AI parsed intent:")
        print(_json.dumps(ai_intent, indent=2, sort_keys=True, ensure_ascii=False))
    except Exception:
        print(f"AI parsed intent: {ai_intent}")

    # Fallback to the original regex parser if AI is unavailable/invalid.
    if not ai_intent or not ai_intent.get("signal"):
        info = extract_order_info(text)

        # Prefer a symbol that actually exists in MT5, if present in the text.
        info_symbol = info.get('symbol')
        normalized_symbol = normalize_symbol(info_symbol)
        resolved_symbol = resolve_mt5_symbol_from_text(text) or normalized_symbol
        
        if resolved_symbol:
            info['symbol'] = resolved_symbol
        else:
            print("Could not resolve a tradable MT5 symbol from message; skipping.")
            return

        # The MT5 market execution path uses live tick prices, so we can proceed even
        # when the signal doesn't include an explicit entry price.
        if 'order_price' not in info:
            print("Order price not found (market execution uses tick price).")

        tp_values = [value for key, value in info.items() if key.startswith("tp")]
        if not tp_values:
            print("No TP targets found!")
            return

        volume_per_tp = float(lot_size) / len(tp_values)
        i = 1
        msg = []

        for tp in tp_values:
            ticket = place_market_order(
                info['symbol'],
                info['order_type'],
                float(format(float(volume_per_tp), '.2f')),
                float(info['sl']),
                float(tp),
            )
            entry_label = info.get('order_price', 'MKT')
            if ticket:
                print(
                    f"Placed order for {float(format(float(volume_per_tp), '.2f'))} lots of {info['symbol']} "
                    f"at {info['order_type']} {entry_label} with SL {info['sl']} and TP {tp}"
                )
                msg.append(
                    f"TP {i}: Placed order for {float(format(float(volume_per_tp), '.2f'))} lots of {info['symbol']} "
                    f"at {info['order_type']} {entry_label} with SL {info['sl']} and TP {tp}"
                )
            else:
                msg.append(f"TP {i}: Failed placing order for {info['symbol']} (see logs)")
            i += 1

        _safe_notify(context, "\n".join(msg))
        return

    symbol = normalize_symbol(ai_intent.get("symbol"))
    side = ai_intent.get("side")
    order_kind = ai_intent.get("order_kind")
    pending_type = ai_intent.get("pending_type")
    entry = ai_intent.get("entry")
    sl = ai_intent.get("sl")
    tps = ai_intent.get("tps") or []

    lot_from_ai = ai_intent.get("lot")
    base_lot = float(lot_from_ai) if isinstance(lot_from_ai, (int, float)) and float(lot_from_ai) > 0 else float(lot_size)

    if not symbol or side not in ("BUY", "SELL"):
        _safe_notify(context, f"AI parse missing symbol/side. Reason: {ai_intent.get('reason', '')}")
        return
    if sl is None or not tps:
        _safe_notify(context, f"AI parse missing SL/TPs. Reason: {ai_intent.get('reason', '')}")
        return

    volume_per_tp = float(base_lot) / len(tps)
    volume_per_tp = float(format(volume_per_tp, '.2f'))

    i = 1
    msg = []
    for tp in tps:
        if order_kind == "PENDING":
            if entry is None or pending_type not in ("LIMIT", "STOP"):
                _safe_notify(context, f"Pending order missing entry/type. Reason: {ai_intent.get('reason', '')}")
                return
            ticket = place_pending_order(symbol, side, pending_type, volume_per_tp, float(entry), float(sl), float(tp))
            if ticket:
                msg.append(
                    f"TP {i}: Placed {side} {pending_type} pending for {volume_per_tp} lots of {symbol} "
                    f"at entry {entry} with SL {sl} and TP {tp}"
                )
            else:
                msg.append(f"TP {i}: Failed placing pending order for {symbol} (see logs)")
        else:
            ticket = place_market_order(symbol, side, volume_per_tp, float(sl), float(tp))
            if ticket:
                msg.append(
                    f"TP {i}: Placed MARKET {side} for {volume_per_tp} lots of {symbol} with SL {sl} and TP {tp}"
                )
            else:
                msg.append(f"TP {i}: Failed placing market order for {symbol} (see logs)")
        i += 1

    _safe_notify(context, "\n".join(msg))
 

def extract_order_info(text: str) -> dict:
    results = {}
    # Symbol extraction (avoid matching ordinary words like "Risky").
    # Prefer:
    # 1) token after BUY/SELL
    # 2) 6-letter FX style symbols (XAUUSD)
    # 3) symbols with digits/underscore/dot (USTEC_X100m)
    symbol = None
    after_side = re.search(r'\b(?:BUY|SELL)\b\s+(?:NOW\s+)?([A-Z][A-Z0-9_.]{2,15})\b', text, re.IGNORECASE)
    if after_side:
        cand = after_side.group(1)
        if re.fullmatch(r'[A-Z]{6}', cand, re.IGNORECASE) or re.search(r'[0-9_.]', cand):
            symbol = cand

    if not symbol:
        candidates = re.findall(r'\b[A-Z][A-Z0-9_.]{2,15}\b', text, flags=re.IGNORECASE)
        stop = {"BUY", "SELL", "NOW", "SL", "TP", "ENTRY", "TAKE", "EVERY", "PIPS", "RISK", "TRADE", "DISCLAIMER"}
        for cand in candidates:
            up = cand.upper()
            if up in stop:
                continue
            if re.fullmatch(r'[A-Z]{6}', cand, re.IGNORECASE) or re.search(r'[0-9_.]', cand):
                symbol = cand
                break

    if symbol:
        results['symbol'] = symbol

    order_type_match = re.search(r'(BUY|SELL)', text, re.IGNORECASE)
    if order_type_match:
        results['order_type'] = order_type_match.group(1).upper()

    price_match = re.search(
        r'(BUY|SELL)(?:\s+NOW)?\s+([\d\.]+)(?:\/[\d\.]+)?|Entry\s*:\s*([\d\.]+)', text, re.IGNORECASE
    )

    if price_match:
        # Check if the first group (BUY/SELL price) matched or the third group (Entry price)
        if price_match.group(2):
            results['order_price'] = float(format(float(price_match.group(2)), ".3f"))
        elif price_match.group(3):
            results['order_price'] = float(format(float(price_match.group(3)), ".3f"))

    sl_match = re.search(r'SL\s*:?\s*([\d:,\']+)(?=\D|$)', text, re.IGNORECASE)
    if sl_match:
        sl_value = sl_match.group(1).replace(",", "").replace("'", ".").replace(":", ".")
        results['sl'] = float(format(float(sl_value), ".3f"))

    tpReplacePattern = r'(Tp\d*\s*);'
    def replace_semicolon(match):
        return match.group(0).replace(';', ':')

    updated_text = re.sub(tpReplacePattern, replace_semicolon, text, flags=re.IGNORECASE)
    tp_matches = re.findall(r'TP\w?\s*:?\s*([\d:,\']+)(?=\D|$)', updated_text, re.IGNORECASE)
    for index, tp_value in enumerate(tp_matches, start=1):
        tp_value = tp_value.replace(",", "").replace("'", ".").replace(":", ".")
        key = f"tp{index}"
        results[key] = float(format(float(tp_value), ".3f"))

    return results

def extract_symbol(text: str) -> str:
    symbol_match = re.search(r'\b([A-Z][A-Z0-9_.]{2,15})\b', text, re.IGNORECASE)
    return symbol_match.group(1) if symbol_match else None


def resolve_mt5_symbol_from_text(text: str) -> str | None:
    """Pick the first symbol-like token that MT5 recognizes."""
    candidates = re.findall(r'\b[A-Z][A-Z0-9_.]{2,15}\b', text or "", flags=re.IGNORECASE)
    seen = set()
    for cand in candidates:
        sym = normalize_symbol(cand)
        if sym in seen:
            continue
        seen.add(sym)
        # Only consider realistic symbols.
        if not (re.fullmatch(r'[A-Z]{6}', sym) or re.search(r'[0-9_.]', sym)):
            continue

        try:
            info = mt5.symbol_info(sym)
            if info is None:
                # Some brokers require selecting the symbol before it becomes available.
                if mt5.symbol_select(sym, True):
                    info = mt5.symbol_info(sym)
            if info is not None:
                return sym
        except Exception:
            continue
    return None


def mt5_symbol_candidates_for_message(text: str) -> list[str]:
    """Return a small list of plausible MT5 symbols for this message.

    This is used to constrain Gemini so it returns an exact MT5 symbol string.
    """
    candidates = re.findall(r'\b[A-Z][A-Z0-9_.]{2,20}\b', text or "", flags=re.IGNORECASE)
    bases: set[str] = set()
    for cand in candidates:
        up = normalize_symbol(cand)
        # Only consider realistic symbols.
        if re.fullmatch(r'[A-Z]{6}', up) or re.search(r'[0-9_.]', up):
            bases.add(up)
        # If it starts with 6 letters, also consider the first 6 (e.g., XAUUSDf -> XAUUSD).
        if len(up) >= 6 and re.fullmatch(r'[A-Z]{6}', up[:6]):
            bases.add(up[:6])

    if not bases:
        return []

    found: list[str] = []
    seen: set[str] = set()
    for base in list(bases)[:10]:
        try:
            # group supports wildcard patterns on many installs (e.g. "*XAUUSD*").
            syms = mt5.symbols_get(group=f"*{base}*")
        except Exception:
            syms = None

        if not syms:
            # Try direct select/info.
            try:
                if mt5.symbol_select(base, True) and mt5.symbol_info(base) is not None:
                    if base not in seen:
                        found.append(base)
                        seen.add(base)
            except Exception:
                pass
            continue

        for info in syms:
            name = getattr(info, 'name', None)
            if not isinstance(name, str):
                continue
            upname = name.upper()
            if upname in seen:
                continue
            # Keep list tight; we just want plausible matches.
            if base in upname or upname.startswith(base):
                found.append(name)
                seen.add(upname)
            if len(found) >= 50:
                break
        if len(found) >= 50:
            break

    return found

def normalize_symbol(symbol: str) -> str:
    """Normalize common trading abbreviations to standard MT5 symbols."""
    if not symbol:
        return symbol
        
    mapping = {
        'UJ': 'USDJPY',
        'GU': 'GBPUSD',
        'EU': 'EURUSD',
        'EJ': 'EURJPY',
        'GJ': 'GBPJPY',
        'GOLD': 'XAUUSD',
        'NAS': 'USTEC', # Common variation, though this can vary by broker
        'US30': 'DJI',  # Common variation
    }
    
    # Check case-insensitive
    upper_sym = symbol.upper()
    if upper_sym in mapping:
        return mapping[upper_sym]
        
    return symbol


def _iter_filling_modes(preferred: int | None = None):
    modes = [preferred, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
    seen = set()
    for m in modes:
        if m is None:
            continue
        if m in seen:
            continue
        seen.add(m)
        yield m


def send_order_with_supported_filling(request: dict, symbol: str):
    """Send MT5 order trying supported filling modes.

    Brokers can reject filling modes per-symbol (common: 'Unsupported filling mode').
    This retries with alternative modes when that specific issue is detected.
    """
    preferred = None
    try:
        info = mt5.symbol_info(symbol)
        if info is not None:
            preferred = getattr(info, 'filling_mode', None)
    except Exception:
        preferred = None

    last_result = None
    for mode in _iter_filling_modes(preferred):
        req = dict(request)
        req['type_filling'] = mode
        result = mt5.order_send(req)
        last_result = result

        if result is None:
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result

        comment = (getattr(result, 'comment', '') or '').lower()
        if 'unsupported' in comment and 'filling' in comment:
            continue

        return result

    return last_result

def place_market_order(symbol, action, volume, sl, tp):
    symbol = normalize_symbol(symbol)
    if action == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            print(f"Could not fetch tick data for symbol: {symbol}")
            return
        price = tick_info.ask
    elif action == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            print(f"Could not fetch tick data for symbol: {symbol}")
            return
        price = tick_info.bid
    else:
        print(f"Unknown action: {action}")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 99999,
        "comment": "python script op",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = send_order_with_supported_filling(request, symbol)

    if result is None:
        print("Failed to send order. No response received.")
        error = mt5.last_error()
        print("Error in order_send(): ", error)
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to send order. Error: {result.comment}")
        return
    print(f"Order successfully placed with ticket {result.order}")
    return result.order


def place_pending_order(symbol, action, pending_kind, volume, price, sl, tp):
    symbol = normalize_symbol(symbol)
    action = action.upper()
    pending_kind = pending_kind.upper()

    if action == "BUY" and pending_kind == "LIMIT":
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    elif action == "SELL" and pending_kind == "LIMIT":
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
    elif action == "BUY" and pending_kind == "STOP":
        order_type = mt5.ORDER_TYPE_BUY_STOP
    elif action == "SELL" and pending_kind == "STOP":
        order_type = mt5.ORDER_TYPE_SELL_STOP
    else:
        print(f"Unknown pending order type: {action} {pending_kind}")
        return

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 99999,
        "comment": "python script op",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = send_order_with_supported_filling(request, symbol)
    if result is None:
        print("Failed to send pending order. No response received.")
        error = mt5.last_error()
        print("Error in order_send(): ", error)
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to send pending order. Error: {result.comment}")
        return

    print(f"Pending order successfully placed with ticket {result.order}")
    return result.order

def close_half_and_update_sl(symbol: str, new_sl: float):
    symbol = normalize_symbol(symbol)
    positions = mt5.positions_get(symbol=symbol)
    
    if positions:
        for position in positions:
            # Check if this position has already been partially closed
            if "python script cl" not in position.comment:
                half_volume = math.floor(position.volume * 50) / 100  # Round down to 2 decimal places
                
                if half_volume > 0:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        print(f"No tick data for {symbol}; cannot close half.")
                        continue
                    # Close half of this position
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": half_volume,
                        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "position": position.ticket,
                        "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
                        "deviation": 20,
                        "magic": 100,
                        "comment": "python script cl",
                        "type_time": mt5.ORDER_TIME_GTC,
                    }
                    
                    close_result = send_order_with_supported_filling(close_request, symbol)
                    if close_result is None:
                        print(f"Failed to close half of position {position.ticket}. No response received.")
                        error = mt5.last_error()
                        print("Error in order_send(): ", error)
                        continue
                    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Failed to close half of position {position.ticket}. Error: {close_result.comment}")
                    else:
                        print(f"Successfully closed {half_volume} lots of position {position.ticket} for {symbol}")
                    
                    # Update the comment for the remaining position to indicate it has been partially closed
                    modify_comment_request = {
                        "action": mt5.TRADE_ACTION_MODIFY,
                        "position": position.ticket,
                        "comment": "python script close half"
                    }
                    modify_comment_result = mt5.order_send(modify_comment_request)
                    if modify_comment_result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Failed to update comment for position {position.ticket}. Error: {modify_comment_result.comment}")
                    else:
                        print(f"Successfully updated comment for position {position.ticket}")
            else:
                print(f"Position {position.ticket} has already been partially closed. Skipping.")
            
            # Update stop loss for all positions, regardless of whether they were just closed or not
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": position.tp
            }
            
            modify_result = mt5.order_send(modify_request)
            if modify_result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to update stop loss for position {position.ticket}. Error: {modify_result.comment}")
            else:
                print(f"Successfully updated stop loss for position {position.ticket} to {new_sl}")
        
        print(f"Finished closing half of eligible positions and updating stop losses for {symbol}")
    else:
        print(f"No open positions found for {symbol}")

def run_bot():
    retry_attempts = 0
    max_retries = 5
    backoff_time = 10  # Initial backoff time
    cooldown_time = 300  # 5 minutes cooldown after max retries

    while True:
        try:
            # Create Updater object and pass in the bot's token.
            updater = Updater(token, use_context=True)
    
            dp = updater.dispatcher
            dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
            dp.add_error_handler(lambda update, context: logging.exception("Telegram handler error", exc_info=context.error))

            # Start the bot
            updater.start_polling()
            updater.idle()

        except telegram.error.NetworkError as e:
            logging.error(f"Network error encountered: {e}. Retrying in {backoff_time} seconds...")
            retry_attempts += 1
            updater.stop()  # Stop the updater to force it to restart

            if retry_attempts > max_retries:
                logging.error(f"Max retry attempts reached. Pausing for {cooldown_time // 60} minutes.")
                time.sleep(cooldown_time)  # Pause for 5 minutes
                retry_attempts = 0  # Reset retry attempts after cooldown

            time.sleep(backoff_time)
            continue  # Retry after waiting
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Retrying in 10 seconds...")
            updater.stop()  # Stop the updater on any other error as well
            time.sleep(10)
            continue

if __name__ == '__main__':
    initialize_bot()
    run_bot()
