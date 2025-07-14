# In backtester.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from shared_logic import fetch_full_historical_data, calculate_indicators, evaluate_buy_filters

def run_backtest(
    symbol: str, 
    asset_type: str, 
    timeframe: str, 
    full_config: Dict[str, Any], 
    preloaded_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    
    params = full_config.get('strategy_params', {})
    backtest_settings = full_config.get('backtest_settings', {})
    
    initial_capital = backtest_settings.get('initial_capital', 10000)
    commission_pct = backtest_settings.get('commission_pct', 0.0)
    
    # This function no longer needs to log, the optimizer will handle it.
    # logging.info(f"--- Running NON-REPAINTING backtest for {symbol} ---")

    if preloaded_data is not None:
        # logging.info("Using preloaded data source.")
        historical_df = preloaded_data
    else:
        logging.info("No preloaded data found. Fetching data from source (Unit Test Mode).")
        start_date_str = full_config.get('strategy_start_timestamp')
        end_date_str = full_config.get('strategy_end_timestamp') 
        
        lookback_periods = [params.get('long_window', 0), params.get('price_ma_filter_period', 0), params.get('rsi_period', 0), params.get('volume_ma_period', 0), params.get('atr_period', 0), params.get('custom_slope_ma_period', 0)]
        longest_lookback_bars = max(lookback_periods) if lookback_periods else 0
        warmup_period_in_bars = int(longest_lookback_bars * 1.5)
        timeframe_unit = 'D'
        if 'm' in timeframe: timeframe_unit = 'min'
        elif 'h' in timeframe: timeframe_unit = 'h'
        elif 'w' in timeframe: timeframe_unit = 'W'
        warmup_timedelta = pd.to_timedelta(warmup_period_in_bars, unit=timeframe_unit)
        
        df = fetch_full_historical_data(asset_type, symbol, timeframe, start_date_str, end_date_str, warmup_timedelta)
        if df is None or df.empty: return {"error": f"Failed to fetch historical data."}
        
        df_indicators = calculate_indicators(df, params)
        
        try:
            first_valid_ma_index = df_indicators[['ma_short', 'ma_long']].dropna().index.min()
            if pd.isna(first_valid_ma_index):
                return {"error": "Not enough data for indicator warmup."}
        except KeyError:
             return {"error": f"Not enough data for indicator warmup."}
            
        user_start_ts = pd.to_datetime(start_date_str)
        actual_start_ts = max(user_start_ts, first_valid_ma_index)
        end_ts = pd.to_datetime(end_date_str) if end_date_str else df_indicators.index[-1]
        historical_df = df_indicators[(df_indicators.index >= actual_start_ts) & (df_indicators.index <= end_ts)].copy()

    if len(historical_df) < 3: 
        return {"error": f"Not enough data for backtest after slicing."}

    equity, adjusted_equity, size_factor, trades = initial_capital, initial_capital, 1.0, []
    equity_curve_data = {historical_df.index[0]: initial_capital}
    in_position, entry_price, entry_size_usd, entry_timestamp, entry_bar_index, stop_loss_level, has_dipped_deep = False, 0.0, 0.0, None, -1, 0.0, False
    crossover_events = 0
    
    # --- START OF DEEP DIAGNOSTIC CHANGES ---
    debug_failed_filters = defaultdict(int)
    first_failed_buy_event_details = None # Will store the details of the first failed buy signal
    # --- END OF DEEP DIAGNOSTIC CHANGES ---


    for i in range(2, len(historical_df)):
        lookbehind_candle, signal_candle, action_candle = historical_df.iloc[i-2], historical_df.iloc[i-1], historical_df.iloc[i]
        
        # ... (Existing in_position logic is unchanged)
        if in_position:
            exit_price, reason = 0.0, ""
            if params.get('use_stop_loss') and action_candle['low'] <= stop_loss_level:
                exit_price, reason = stop_loss_level, "Stop Loss"
            if exit_price == 0.0 and params.get('use_recovery_exit'):
                if not has_dipped_deep and (entry_price - signal_candle['close']) / entry_price * 100 >= params.get('dip_pct_trigger', 2.0):
                    has_dipped_deep = True
                if has_dipped_deep and abs(signal_candle['close'] - entry_price) / entry_price * 100 <= params.get('recovery_pct_threshold', 0.0):
                    exit_price, reason = action_candle['open'], "Recovery Exit"
            if exit_price == 0.0:
                slope_sell = False; rsi_sell = False
                if signal_candle.get('ma_short_slope', 0) <= 0: slope_sell = True
                if params.get('use_rsi_filter') and signal_candle.get('rsi', 0) > params.get('rsi_sell_threshold', 100): rsi_sell = True
                bars_held = (i - 1) - entry_bar_index
                pass_hold = not params.get('use_hold_duration_filter') or (bars_held >= params.get('min_hold_bars', 0))
                if (slope_sell or rsi_sell) and pass_hold: exit_price, reason = action_candle['open'], "Sell Signal"
            if exit_price > 0.0:
                exit_value = (entry_size_usd / entry_price) * exit_price; commission = (entry_size_usd + exit_value) * (commission_pct / 100.0)
                profit = (exit_value - entry_size_usd) - commission; equity += profit
                trades.append({"entry_ts": entry_timestamp, "exit_ts": action_candle.name, "entry_price": entry_price, "exit_price": exit_price, "profit": profit, "commission": commission, "reason": reason, "entry_size_usd": entry_size_usd})
                equity_curve_data[action_candle.name] = equity
                if params.get('use_size_down'): size_factor = (params.get('size_down_pct', 50) / 100.0) if profit < 0 else 1.0
                if params.get('use_fake_loss'): adjusted_equity = equity - (profit * (params.get('withdrawal_pct', 80) / 100.0) if profit > 0 else 0)
                else: adjusted_equity = equity
                in_position = False; continue

        if not in_position:
            ma_short_lookbehind, ma_long_lookbehind = lookbehind_candle.get('ma_short'), lookbehind_candle.get('ma_long')
            ma_short_signal, ma_long_signal = signal_candle.get('ma_short'), signal_candle.get('ma_long')
            is_golden_cross = False
            if pd.notna([ma_short_lookbehind, ma_long_lookbehind, ma_short_signal, ma_long_signal]).all():
                is_golden_cross = ma_short_lookbehind <= ma_long_lookbehind and ma_short_signal > ma_long_signal
            
            if is_golden_cross:
                crossover_events += 1
                all_filters_passed, failed_filters = evaluate_buy_filters(signal_candle, params)
                pass_cooldown = True
                if params.get('use_cooldown_filter') and trades:
                    bars_since_last_trade = (i - 1) - historical_df.index.get_loc(trades[-1]['exit_ts'])
                    if bars_since_last_trade < params.get('cooldown_bars', 0): pass_cooldown = False
                if not pass_cooldown: failed_filters.append('Cooldown Period')
                
                if not all_filters_passed or not pass_cooldown:
                    for reason_failed in set(failed_filters): debug_failed_filters[reason_failed] += 1
                    
                    # --- START OF DEEP DIAGNOSTIC CAPTURE ---
                    # If this is the first failed buy signal we've seen, record everything about it.
                    if first_failed_buy_event_details is None:
                        first_failed_buy_event_details = {
                            "timestamp": signal_candle.name.isoformat(),
                            "failed_filters": failed_filters,
                            "candle_data": {
                                "close": signal_candle.get('close'),
                                "rsi": f"{signal_candle.get('rsi'):.2f}" if pd.notna(signal_candle.get('rsi')) else "N/A",
                                "filter_ma": f"{signal_candle.get('filter_ma'):.2f}" if pd.notna(signal_candle.get('filter_ma')) else "N/A",
                                "ma_short_slope": f"{signal_candle.get('ma_short_slope'):.4f}" if pd.notna(signal_candle.get('ma_short_slope')) else "N/A",
                                "volume": signal_candle.get('volume')
                            },
                            "relevant_params": {
                                "rsi_buy_threshold": params.get('rsi_buy_threshold') if params.get('use_rsi_filter') else "Disabled",
                                "price_ma_filter_enabled": params.get('use_price_ma_filter'),
                                "slope_threshold": params.get('slope_threshold') if params.get('slope_filter_mode') != "Nonaktif" else "Disabled"
                            }
                        }
                    # --- END OF DEEP DIAGNOSTIC CAPTURE ---

                else: # This block is `if all_filters_passed and pass_cooldown:`
                    in_position, entry_price, entry_timestamp, entry_bar_index, has_dipped_deep = True, action_candle['open'], action_candle.name, i - 1, False
                    equity_to_use = adjusted_equity if params.get('use_fake_loss') else equity
                    final_pos_pct = (params.get('base_position_pct', 95) * params.get('leverage_multiplier', 1.0))
                    entry_size_usd = equity_to_use * (final_pos_pct / 100.0) * size_factor
                    if params.get('use_stop_loss'): stop_loss_level = entry_price * (1 - params.get('stop_loss_pct', 10) / 100.0)

    # ... (Final trade closure logic is unchanged)
    if in_position:
        last_candle = historical_df.iloc[-1]; exit_price = last_candle['close']
        exit_value = (entry_size_usd / entry_price) * exit_price; commission = (entry_size_usd + exit_value) * (commission_pct / 100.0)
        profit = (exit_value - entry_size_usd) - commission; equity += profit
        trades.append({"entry_ts": entry_timestamp, "exit_ts": last_candle.name, "entry_price": entry_price, "exit_price": exit_price, "profit": profit, "commission": commission, "reason": "End of Backtest", "entry_size_usd": entry_size_usd})
        equity_curve_data[last_candle.name] = equity

    # --- START OF DEEP DIAGNOSTIC RETURN ---
    debug_info = {
        "crossover_events_found": crossover_events, 
        "failed_filter_counts": dict(debug_failed_filters),
        "first_failed_buy_event_details": first_failed_buy_event_details # Add our new detailed log
    }
    
    if not trades:
        return {
            "message": "No trades were executed.", 
            "metrics": {},
            "debug_info": debug_info,
            "historical_data": historical_df.to_dict('index')
        }
    # --- END OF DEEP DIAGNOSTIC RETURN ---
    
    equity_curve = pd.Series(equity_curve_data).sort_index()
    peak_equity = equity_curve.cummax()
    drawdown_series = peak_equity - equity_curve
    max_drawdown = drawdown_series.max()
    buy_hold_start_price = historical_df['open'].iloc[0]
    buy_hold_units = initial_capital / buy_hold_start_price
    buy_and_hold_equity = historical_df['close'] * buy_hold_units
    final_equity, net_profit = equity, equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100 if initial_capital > 0 else 0
    trade_df = pd.DataFrame(trades)
    total_trades = len(trade_df)
    win_rate = (len(trade_df[trade_df['profit'] > 0]) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit, gross_loss = trade_df[trade_df['profit'] > 0]['profit'].sum(), abs(trade_df[trade_df['profit'] <= 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {"metrics": {"net_profit_pct": net_profit_pct, "total_trades": total_trades, "win_rate": win_rate, "profit_factor": profit_factor, "max_drawdown": max_drawdown, "max_drawdown_pct": (max_drawdown / peak_equity.max()) * 100 if peak_equity.max() > 0 else 0}, "trades": trade_df.to_dict('records'), "equity_curve": equity_curve.to_dict(), "buy_and_hold_equity": buy_and_hold_equity.to_dict(), "drawdown_series": drawdown_series.to_dict(), "historical_data": historical_df.to_dict('index'), "debug_info": debug_info}