import os
import json
import time
import logging
import schedule
import requests
import pandas as pd
from typing import Dict, Any
from shared_logic import fetch_data, calculate_indicators, evaluate_buy_filters, load_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_json(filepath: str, data: Dict):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def send_telegram_alert(message: str):
    config = load_json('config.json', {})
    if not config.get('telegram', {}).get('enabled', False):
        logging.info(f"Telegram disabled. Would have sent: {message}")
        return
    
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logging.error("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables not set.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Successfully sent Telegram alert.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Telegram alert: {e}")

def main_job():
    logging.info("--- Alerter: Starting new scan cycle ---")
    alerts = load_json('alerts.json', default_value=[])
    state = load_json('state.json', default_value={})

    if not alerts:
        logging.info("Alerter: No active alerts found in alerts.json. Sleeping.")
        return

    for alert in alerts:
        alert_id = alert.get('id')
        if not alert_id: continue

        params = alert.get('params', {})
        symbol = alert.get('symbol')
        timeframe = alert.get('timeframe')
        asset_type = alert.get('asset_type')
        
        logging.info(f"Alerter: Checking alert '{alert_id}' for {symbol} ({timeframe})")
        
        try:
            data = fetch_data(asset_type, symbol, timeframe, params)
            if data is None or data.empty:
                logging.warning(f"Alerter: Skipping {symbol} due to no data.")
                continue
            
            df = calculate_indicators(data, params)
            if len(df) < 3: 
                logging.warning(f"Alerter: Insufficient data for {symbol} after indicator calculation.")
                continue

            lookbehind_candle = df.iloc[-3]
            signal_candle = df.iloc[-2]
            
            alert_state = state.get(alert_id, {'in_position': False})

            if not alert_state.get('in_position'):
                is_golden_cross = lookbehind_candle.get('ma_short', 0) <= lookbehind_candle.get('ma_long', 1) and \
                                  signal_candle.get('ma_short', 0) > signal_candle.get('ma_long', 1)
                if is_golden_cross:
                    all_filters_passed, _ = evaluate_buy_filters(signal_candle, params)
                    if all_filters_passed:
                        rsi_value = signal_candle.get('rsi')
                        rsi_str = f"{rsi_value:.2f}" if rsi_value is not None and pd.notna(rsi_value) else "N/A"
                        message = (f"ðŸš€ *LIVE BUY SIGNAL* ðŸš€\n\n"
                                   f"*Alert ID:* `{alert_id}`\n"
                                   f"*Asset:* `{symbol}` ({timeframe})\n"
                                   f"*Signal Price:* `{signal_candle['close']:.4f}`\n"
                                   f"*RSI:* `{rsi_str}`")
                        send_telegram_alert(message)
                        
                        state[alert_id] = {'in_position': True, 'entry_price': signal_candle['close'], 'has_dipped_deep': False, 'entry_bar_index': len(df)-2}
                        save_json('state.json', state)
            else:
                entry_price = alert_state.get('entry_price', 0)
                if entry_price == 0: continue

                exit_triggered, reason = False, ""
                
                is_recovery_exit_triggered = False
                if params.get('use_recovery_exit'):
                    has_dipped = alert_state.get('has_dipped_deep', False)
                    if not has_dipped and (entry_price - signal_candle['close']) / entry_price * 100 >= params.get('dip_pct_trigger', 2.0):
                        alert_state['has_dipped_deep'] = True
                        has_dipped = True
                    if has_dipped and abs(signal_candle['close'] - entry_price) / entry_price * 100 <= params.get('recovery_pct_threshold', 0.0):
                        is_recovery_exit_triggered = True

                if is_recovery_exit_triggered:
                    exit_triggered, reason = True, "Recovery Exit"
                else:
                    # --- MODIFICATION START: Combined MA Slope and RSI sell logic ---
                    bars_held = (len(df)-2) - alert_state.get('entry_bar_index', len(df)-2)
                    pass_hold_filter = not params.get('use_hold_duration_filter') or (bars_held >= params.get('min_hold_bars', 0))

                    if pass_hold_filter:
                        is_ma_slope_sell = signal_candle.get('ma_short_slope', 0) <= 0
                        
                        passes_rsi_sell_filter = not params.get('use_rsi_filter') or \
                                                 (signal_candle.get('rsi', 0) >= params.get('rsi_sell_threshold', 100))

                        if is_ma_slope_sell and passes_rsi_sell_filter:
                            exit_triggered, reason = True, "Sell Signal"
                    # --- MODIFICATION END ---
                
                if exit_triggered:
                    profit_pct = (signal_candle['close'] - entry_price) / entry_price * 100
                    rsi_value = signal_candle.get('rsi')
                    rsi_str = f"{rsi_value:.2f}" if rsi_value is not None and pd.notna(rsi_value) else "N/A"
                    message = (f"ðŸ›‘ *LIVE SELL SIGNAL ({reason})* ðŸ›‘\n\n"
                               f"*Alert ID:* `{alert_id}`\n*Asset:* `{symbol}`\n"
                               f"*Exit Price:* `{signal_candle['close']:.4f}`\n*Entry Price:* `{entry_price:.4f}`\n"
                               f"*P/L:* `{profit_pct:.2f}%`\n"
                               f"*RSI:* `{rsi_str}`")
                    send_telegram_alert(message)
                    
                    state[alert_id] = {'in_position': False}
                    save_json('state.json', state)
                else:
                    save_json('state.json', state)

        except Exception as e:
            logging.error(f"Alerter: Error processing alert {alert_id}: {e}", exc_info=True)
    
    logging.info("--- Alerter: Scan cycle finished ---")

if __name__ == "__main__":
    logging.info("Alerter script starting...")
    config = load_json('config.json', {})
    interval = config.get('scan_interval_minutes', 15)
    
    main_job() 
    schedule.every(interval).minutes.do(main_job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)