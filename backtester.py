import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from shared_logic import fetch_full_historical_data, calculate_indicators, evaluate_buy_filters

def run_backtest(symbol: str, asset_type: str, timeframe: str, full_config: Dict[str, Any]) -> Dict[str, Any]:
    params = full_config.get('strategy_params', {})
    backtest_settings = full_config.get('backtest_settings', {})
    start_date_str = full_config.get('strategy_start_timestamp')
    initial_capital = backtest_settings.get('initial_capital', 10000)
    commission_pct = backtest_settings.get('commission_pct', 0.0)
    logging.info(f"--- Running NON-REPAINTING backtest for {symbol} ---")
    lookback_periods = [params.get('long_window', 0), params.get('price_ma_filter_period', 0) if params.get('use_price_ma_filter') else 0, params.get('rsi_period', 0) if params.get('use_rsi_filter') else 0, params.get('volume_ma_period', 0) if params.get('use_volume_filter') else 0, params.get('atr_period', 0) if params.get('use_volatility_filter') else 0, params.get('custom_slope_ma_period', 0) if params.get('slope_filter_mode') == 'Gunakan MA Kustom' else 0]
    longest_lookback_bars = max(lookback_periods) if lookback_periods else 0
    warmup_period_in_bars = int(longest_lookback_bars * 1.5)
    timeframe_unit = 'D'
    if 'm' in timeframe: timeframe_unit = 'min'
    elif 'h' in timeframe: timeframe_unit = 'H'
    elif 'w' in timeframe: timeframe_unit = 'W'
    warmup_timedelta = pd.to_timedelta(warmup_period_in_bars, unit=timeframe_unit)
    df = fetch_full_historical_data(asset_type, symbol, timeframe, start_date_str, warmup_timedelta)
    if df is None or df.empty: return {"error": f"Failed to fetch deep historical data."}
    df_indicators = calculate_indicators(df, params)
    first_valid_ma_index = df_indicators[['ma_short', 'ma_long']].dropna().index.min()
    if pd.isna(first_valid_ma_index): return {"error": "Could not calculate moving averages. Check data or indicator periods."}
    user_start_ts = pd.to_datetime(start_date_str)
    actual_start_ts = max(user_start_ts, first_valid_ma_index)
    historical_df = df_indicators[df_indicators.index >= actual_start_ts].copy()
    if len(historical_df) < 3: return {"error": f"Not enough data for backtest after start date and indicator warmup."}
    equity, adjusted_equity, size_factor, trades = initial_capital, initial_capital, 1.0, []
    equity_curve_data = {historical_df.index[0]: initial_capital}
    in_position, entry_price, entry_size_usd, entry_timestamp, entry_bar_index, stop_loss_level, has_dipped_deep = False, 0.0, 0.0, None, -1, 0.0, False
    debug_failed_filters, debug_log, crossover_events = defaultdict(int), [], 0

    for i in range(2, len(historical_df)):
        lookbehind_candle, signal_candle, action_candle = historical_df.iloc[i-2], historical_df.iloc[i-1], historical_df.iloc[i]
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
                ma_short_slope = signal_candle.get('ma_short_slope')
                is_sell_trigger = ma_short_slope <= 0 if pd.notna(ma_short_slope) else False
                pass_rsi_sell = not params.get('use_rsi_filter') or (signal_candle.get('rsi', 100) > params.get('rsi_sell_threshold', 100))
                bars_held = (i - 1) - entry_bar_index
                pass_hold = not params.get('use_hold_duration_filter') or (bars_held >= params.get('min_hold_bars', 0))
                if is_sell_trigger and pass_rsi_sell and pass_hold:
                    exit_price, reason = action_candle['open'], "Sell Signal"
            if exit_price > 0.0:
                exit_value = (entry_size_usd / entry_price) * exit_price
                commission = (entry_size_usd + exit_value) * (commission_pct / 100.0)
                profit = (exit_value - entry_size_usd) - commission
                equity += profit
                trades.append({"entry_ts": entry_timestamp, "exit_ts": action_candle.name, "entry_price": entry_price, "exit_price": exit_price, "profit": profit, "commission": commission, "reason": reason, "entry_size_usd": entry_size_usd})
                equity_curve_data[action_candle.name] = equity
                if params.get('use_size_down'): size_factor = (params.get('size_down_pct', 50) / 100.0) if profit < 0 else 1.0
                if params.get('use_fake_loss'):
                    withdrawal = profit * (params.get('withdrawal_pct', 80) / 100.0) if profit > 0 else 0
                    adjusted_equity = equity - withdrawal
                else: adjusted_equity = equity
                in_position = False
                continue
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
                    last_exit_ts = trades[-1]['exit_ts']
                    if last_exit_ts in historical_df.index:
                        bars_since_last_trade = (i - 1) - historical_df.index.get_loc(last_exit_ts)
                        if bars_since_last_trade < params.get('cooldown_bars', 0): pass_cooldown = False
                if not pass_cooldown: failed_filters.append('Cooldown Period')
                if not all_filters_passed or not pass_cooldown:
                    for reason_failed in set(failed_filters): debug_failed_filters[reason_failed] += 1
                    if len(debug_log) < 20: debug_log.append({"timestamp": signal_candle.name.isoformat(), "failed_filters": ", ".join(failed_filters), "rsi_value": f"{signal_candle.get('rsi'):.2f}" if 'rsi' in signal_candle else "N/A", "rsi_threshold": params.get('rsi_buy_threshold') if params.get('use_rsi_filter') else "N/A"})
                if all_filters_passed and pass_cooldown:
                    in_position, entry_price, entry_timestamp, entry_bar_index, has_dipped_deep = True, action_candle['open'], action_candle.name, i - 1, False
                    equity_to_use = adjusted_equity if params.get('use_fake_loss') else equity
                    final_pos_pct = (params.get('base_position_pct', 95) * params.get('leverage_multiplier', 1.0))
                    entry_size_usd = equity_to_use * (final_pos_pct / 100.0) * size_factor
                    if params.get('use_stop_loss'): stop_loss_level = entry_price * (1 - params.get('stop_loss_pct', 10) / 100.0)

    # --- START OF FINAL FIX ---
    # After loop, check if still in position and close it at the last candle's close price
    if in_position:
        last_candle = historical_df.iloc[-1]
        exit_price = last_candle['close']
        exit_value = (entry_size_usd / entry_price) * exit_price
        commission = (entry_size_usd + exit_value) * (commission_pct / 100.0)
        profit = (exit_value - entry_size_usd) - commission
        equity += profit
        trades.append({"entry_ts": entry_timestamp, "exit_ts": last_candle.name, "entry_price": entry_price, "exit_price": exit_price, "profit": profit, "commission": commission, "reason": "End of Backtest", "entry_size_usd": entry_size_usd})
        equity_curve_data[last_candle.name] = equity
    # --- END OF FINAL FIX ---

    if not trades: return {"message": "No trades were executed.", "debug_info": {"crossover_events_found": crossover_events, "failed_filter_counts": dict(debug_failed_filters), "detailed_log": debug_log}}
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
    return {"metrics": {"net_profit_pct": net_profit_pct, "total_trades": total_trades, "win_rate": win_rate, "profit_factor": profit_factor, "max_drawdown": max_drawdown, "max_drawdown_pct": (max_drawdown / peak_equity.max()) * 100 if peak_equity.max() > 0 else 0}, "trades": trade_df.to_dict('records'), "equity_curve": equity_curve.to_dict(), "buy_and_hold_equity": buy_and_hold_equity.to_dict(), "drawdown_series": drawdown_series.to_dict(), "historical_data": historical_df.to_dict('index')}