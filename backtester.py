# backtester.py


import logging
import pandas as pd
import numpy as np
import numba
from numba import jit
from typing import Dict, Any, List, Optional
from collections import defaultdict
from shared_logic import fetch_full_historical_data, calculate_indicators

# --- NEW: Numba JIT-compiled core backtesting loop for performance ---
@jit(nopython=True)
def _execute_backtest_loop(
    # --- Input Data (as numpy arrays) ---
    timestamps_arr: np.ndarray,
    open_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    buy_signals: np.ndarray,
    sell_signals: np.ndarray,
    rsi_arr: np.ndarray,
    # --- State and Parameters ---
    initial_capital: float,
    commission_pct: float,
    use_stop_loss: bool,
    stop_loss_pct: float,
    use_recovery_exit: bool,
    dip_pct_trigger: float,
    recovery_pct_threshold: float,
    use_cooldown_filter: bool,
    cooldown_bars: int,
    use_hold_duration_filter: bool,
    min_hold_bars: int,
    base_position_pct: float,
    leverage_multiplier: float,
    use_size_down: bool,
    size_down_pct: float,
    use_fake_loss: bool,
    withdrawal_pct: float,
) -> (np.ndarray, np.ndarray, int):
    """
    Core backtest simulation loop, accelerated with Numba.
    This function is path-dependent and handles all state changes.
    """
    n = len(close_arr)
    equity = initial_capital
    adjusted_equity = initial_capital
    size_factor = 1.0
    in_position = False
    entry_price, entry_size_usd, entry_bar_index, stop_loss_level, has_dipped_deep = 0.0, 0.0, -1, 0.0, False

    # Pre-allocate arrays for trade results for performance
    max_trades = len(buy_signals)
    trade_results = np.zeros((max_trades, 8)) # entry_ts_idx, exit_ts_idx, entry_price, exit_price, profit, commission, reason_code, entry_size_usd
    trade_count = 0

    equity_curve_ts = np.zeros(n)
    equity_curve_val = np.zeros(n)
    equity_curve_idx = 0

    last_trade_exit_bar = -cooldown_bars

    # Reason codes for exits
    REASON_SELL_SIGNAL = 1
    REASON_STOP_LOSS = 2
    REASON_RECOVERY_EXIT = 3
    REASON_END_OF_BACKTEST = 4

    for i in range(1, n):
        # ----------------------------------------------------------------------
        # --- 1. HANDLE EXITS FOR CURRENTLY OPEN POSITIONS ---
        # ----------------------------------------------------------------------
        if in_position:
            exit_price, reason_code = 0.0, 0

            # Priority 1: Stop Loss
            if use_stop_loss and low_arr[i] <= stop_loss_level:
                exit_price, reason_code = stop_loss_level, REASON_STOP_LOSS

            # Priority 2: Recovery Exit (based on previous candle's close)
            if exit_price == 0.0 and use_recovery_exit:
                if not has_dipped_deep and (entry_price - close_arr[i-1]) / entry_price >= dip_pct_trigger:
                    has_dipped_deep = True
                if has_dipped_deep and abs(close_arr[i-1] - entry_price) / entry_price <= recovery_pct_threshold:
                    exit_price, reason_code = open_arr[i], REASON_RECOVERY_EXIT

            # Priority 3: Primary Sell Signal
            if exit_price == 0.0:
                bars_held = i - entry_bar_index
                pass_hold_filter = not use_hold_duration_filter or (bars_held >= min_hold_bars)
                if pass_hold_filter and sell_signals[i-1]: # sell signal from previous candle
                    exit_price, reason_code = open_arr[i], REASON_SELL_SIGNAL

            # --- Process The Exit ---
            if exit_price > 0.0:
                exit_value = (entry_size_usd / entry_price) * exit_price
                commission = (entry_size_usd + exit_value) * commission_pct
                profit = (exit_value - entry_size_usd) - commission
                equity += profit

                # Record the trade
                trade_results[trade_count] = [entry_bar_index, i, entry_price, exit_price, profit, commission, reason_code, entry_size_usd]
                trade_count += 1

                # Update equity curve
                equity_curve_ts[equity_curve_idx] = timestamps_arr[i]
                equity_curve_val[equity_curve_idx] = equity
                equity_curve_idx += 1

                # Update risk management state
                if use_size_down and profit < 0: size_factor = size_down_pct
                else: size_factor = 1.0
                if use_fake_loss and profit > 0: adjusted_equity = equity - (profit * withdrawal_pct)
                else: adjusted_equity = equity

                # Reset position state
                in_position = False
                last_trade_exit_bar = i

        # ----------------------------------------------------------------------
        # --- 2. HANDLE ENTRIES BASED ON SIGNALS ---
        # ----------------------------------------------------------------------
        if not in_position and buy_signals[i-1]: # buy signal from previous candle
            # Check Cooldown Period
            is_in_cooldown = use_cooldown_filter and (i - last_trade_exit_bar < cooldown_bars)

            if not is_in_cooldown:
                # --- Process The Entry ---
                in_position = True
                entry_price = open_arr[i]
                entry_bar_index = i
                has_dipped_deep = False

                # Calculate position size
                equity_to_use = adjusted_equity if use_fake_loss else equity
                pos_size_pct = (base_position_pct * leverage_multiplier) / 100.0
                entry_size_usd = equity_to_use * pos_size_pct * size_factor

                if use_stop_loss: stop_loss_level = entry_price * (1 - stop_loss_pct)

    # ----------------------------------------------------------------------
    # --- 3. FINAL CLEANUP: CLOSE ANY OPEN POSITIONS AT THE END ---
    # ----------------------------------------------------------------------
    if in_position:
        exit_price = close_arr[n-1]
        exit_value = (entry_size_usd / entry_price) * exit_price
        commission = (entry_size_usd + exit_value) * commission_pct
        profit = (exit_value - entry_size_usd) - commission
        equity += profit
        trade_results[trade_count] = [entry_bar_index, n-1, entry_price, exit_price, profit, commission, REASON_END_OF_BACKTEST, entry_size_usd]
        trade_count += 1
        equity_curve_ts[equity_curve_idx] = timestamps_arr[n-1]
        equity_curve_val[equity_curve_idx] = equity
        equity_curve_idx += 1

    # Return only the completed trades and equity points
    return trade_results[:trade_count], np.stack((equity_curve_ts[:equity_curve_idx], equity_curve_val[:equity_curve_idx]), axis=-1), equity


def calculate_performance_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
    buy_and_hold_equity: pd.Series
) -> Dict:
    if equity_curve.empty: return {}

    total_trades = len(trades_df)
    if total_trades == 0: return { "total_trades": 0 }

    daily_returns = equity_curve.resample('D').last().ffill().pct_change().dropna()
    sharpe_ratio, sortino_ratio = 0.0, 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        downside_std = daily_returns[daily_returns < 0].std()
        if downside_std > 0:
            sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)

    win_rate = (len(trades_df[trades_df['profit'] > 0]) / total_trades) * 100
    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
    gross_loss = abs(trades_df[trades_df['profit'] <= 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    peak_equity = equity_curve.cummax()
    drawdown_series = peak_equity - equity_curve
    max_drawdown = drawdown_series.max()

    final_equity = equity_curve.iloc[-1]

    return {
        "net_profit_pct": ((final_equity - initial_capital) / initial_capital) * 100,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": (max_drawdown / peak_equity.max()) * 100 if peak_equity.max() > 0 else 0,
        "sharpe_ratio": sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0,
        "sortino_ratio": sortino_ratio if np.isfinite(sortino_ratio) else 0.0,
        "drawdown_series": drawdown_series
    }

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
    commission_pct = backtest_settings.get('commission_pct', 0.0) / 100.0

    if preloaded_data is not None:
        historical_df_raw = preloaded_data
        df_indicators = calculate_indicators(historical_df_raw, params)
        historical_df = df_indicators # Use the full DF for now
    else:
        start_date_str = full_config.get('strategy_start_timestamp')
        end_date_str = full_config.get('strategy_end_timestamp')

        # --- CORRECTED CODE ---
        # First, get all potential lookback values from the params dictionary.
        lookback_values = [params.get(k) for k in ['long_window', 'price_ma_filter_period', 'rsi_period', 'volume_ma_period', 'atr_period', 'custom_slope_ma_period']]

        # Next, create the final list by filtering out any None values and keeping only numbers.
        # This prevents the TypeError when max() is called.
        lookback_periods = [p for p in lookback_values if isinstance(p, (int, float))]

        # Now, calculate the longest lookback safely.
        longest_lookback = max(lookback_periods) if lookback_periods else 0
        warmup_timedelta = pd.to_timedelta(int(longest_lookback * 1.5), unit=timeframe[-1])

        df = fetch_full_historical_data(asset_type, symbol, timeframe, start_date_str, end_date_str, warmup_timedelta)
        if df is None or df.empty: return {"error": "Failed to fetch historical data."}
        
        df_indicators = calculate_indicators(df, params)
        
        # --- FIX START: Robust check for insufficient data ---
        if 'ma_short' not in df_indicators.columns or 'ma_long' not in df_indicators.columns:
            return {"error": "Not enough data for indicator warmup. MA columns are missing."}
        # --- FIX END ---
        
        first_valid_ma = df_indicators[['ma_short', 'ma_long']].dropna().index.min()
        if pd.isna(first_valid_ma): return {"error": "Not enough data for indicator warmup."}

        user_start_ts = pd.to_datetime(start_date_str)
        actual_start_ts = max(user_start_ts, first_valid_ma)
        end_ts = pd.to_datetime(end_date_str) if end_date_str else df_indicators.index[-1]
        historical_df = df_indicators.loc[actual_start_ts:end_ts].copy()

    if len(historical_df) < 3: return {"error": "Not enough data for backtest after slicing."}

    # --- FIX START: Ensure all required columns exist before vectorization ---
    # Create dummy columns if they weren't calculated, to prevent KeyErrors.
    if 'rsi' not in historical_df.columns:
        historical_df['rsi'] = 50 # Use a neutral RSI value
    if 'ma_short_slope' not in historical_df.columns:
        historical_df['ma_short_slope'] = 0.0
    if params.get('use_price_ma_filter') and 'filter_ma' not in historical_df.columns:
         historical_df['filter_ma'] = 0.0
    # --- FIX END ---
    
    # 2. Vectorized Signal Generation
    golden_cross = (historical_df['ma_short'].shift(1) <= historical_df['ma_long'].shift(1)) & \
                   (historical_df['ma_short'] > historical_df['ma_long'])

    buy_filters_passed = pd.Series(True, index=historical_df.index)
    failed_filter_counts = defaultdict(int)
    if params.get('use_price_ma_filter'):
        f = historical_df['close'] >= historical_df['filter_ma']
        failed_filter_counts['Price vs Filter MA'] = len(golden_cross[golden_cross & ~f])
        buy_filters_passed &= f
    if params.get('use_rsi_filter'):
        f = historical_df['rsi'] < params.get('rsi_buy_threshold', 50)
        failed_filter_counts['RSI Buy'] = len(golden_cross[golden_cross & ~f])
        buy_filters_passed &= f
    
    # --- START OF FIX ---
    # Added the vectorized logic for the 'use_price_dist_filter' parameter.
    if params.get('use_price_dist_filter'):
        ma_short = historical_df['ma_short']
        # Calculate the percentage distance between the close price and the short MA.
        # Replace 0 in ma_short with NaN to prevent division by zero errors.
        distance = ( (historical_df['close'] - ma_short).abs() / ma_short.replace(0, np.nan) ) * 100
        # The filter passes if the distance is GREATER than the threshold.
        f = distance > params.get('price_dist_threshold', 0.0)
        # Count how many initial signals failed this specific filter for debugging.
        failed_filter_counts['Price Distance'] = len(golden_cross[golden_cross & ~f])
        # Apply the filter to the master boolean Series.
        buy_filters_passed &= f
    # --- END OF FIX ---

    final_buy_signals = golden_cross & buy_filters_passed
    crossover_events_found = int(golden_cross.sum())

    is_ma_slope_sell = historical_df['ma_short_slope'] <= 0
    passes_rsi_sell_filter = True
    if params.get('use_rsi_filter'):
        passes_rsi_sell_filter = historical_df['rsi'] >= params.get('rsi_sell_threshold', 100)
    final_sell_signals = is_ma_slope_sell & passes_rsi_sell_filter

    np_close = historical_df['close'].to_numpy(dtype=np.float64)

    # 4. Execute the Numba-accelerated Loop
    trade_results_arr, equity_curve_arr, final_equity = _execute_backtest_loop(
        timestamps_arr=historical_df.index.to_numpy(dtype=np.int64),
        open_arr=historical_df['open'].to_numpy(dtype=np.float64),
        low_arr=historical_df['low'].to_numpy(dtype=np.float64),
        close_arr=np_close,
        buy_signals=final_buy_signals.to_numpy(dtype=np.bool_),
        sell_signals=final_sell_signals.to_numpy(dtype=np.bool_),
        rsi_arr=historical_df['rsi'].fillna(50).to_numpy(dtype=np.float64),
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        use_stop_loss=params.get('use_stop_loss', False),
        stop_loss_pct=params.get('stop_loss_pct', 10.0) / 100.0,
        use_recovery_exit=params.get('use_recovery_exit', False),
        dip_pct_trigger=params.get('dip_pct_trigger', 2.0) / 100.0,
        recovery_pct_threshold=params.get('recovery_pct_threshold', 0.1) / 100.0,
        use_cooldown_filter=params.get('use_cooldown_filter', False),
        cooldown_bars=params.get('cooldown_bars', 0),
        use_hold_duration_filter=params.get('use_hold_duration_filter', False),
        min_hold_bars=params.get('min_hold_bars', 0),
        base_position_pct=params.get('base_position_pct', 95.0),
        leverage_multiplier=params.get('leverage_multiplier', 1.0),
        use_size_down=params.get('use_size_down', False),
        size_down_pct=params.get('size_down_pct', 50.0) / 100.0,
        use_fake_loss=params.get('use_fake_loss', False),
        withdrawal_pct=params.get('withdrawal_pct', 80.0) / 100.0,
    )

    debug_info = {"crossover_events_found": crossover_events_found, "failed_filter_counts": dict(failed_filter_counts)}
    if trade_results_arr.shape[0] == 0:
        return {
            "message": "No trades were executed.",
            "metrics": {},
            "debug_info": debug_info,
            "historical_data": historical_df.to_dict('index')
        }

    trades_df = pd.DataFrame(trade_results_arr, columns=["entry_idx", "exit_idx", "entry_price", "exit_price", "profit", "commission", "reason_code", "entry_size_usd"])
    trades_df['entry_ts'] = historical_df.index[trades_df['entry_idx'].astype(int)]
    trades_df['exit_ts'] = historical_df.index[trades_df['exit_idx'].astype(int)]
    # Use .iloc to safely access RSI values by integer location
    trades_df['entry_rsi'] = historical_df['rsi'].iloc[trades_df['entry_idx'].astype(int)].values
    trades_df['exit_rsi'] = historical_df['rsi'].iloc[trades_df['exit_idx'].astype(int) - 1].values

    reason_map = {1: "Sell Signal", 2: "Stop Loss", 3: "Recovery Exit", 4: "End of Backtest"}
    trades_df['reason'] = trades_df['reason_code'].map(reason_map)

    equity_curve = pd.Series(equity_curve_arr[:, 1], index=pd.to_datetime(equity_curve_arr[:, 0], utc=True))
    equity_curve = pd.concat([pd.Series({historical_df.index[0]: initial_capital}), equity_curve]).sort_index()

    buy_hold_equity_series = pd.Series((np_close / np_close[0]) * initial_capital, index=historical_df.index)

    metrics = calculate_performance_metrics(equity_curve, trades_df, initial_capital, buy_hold_equity_series)

    return {
        "metrics": {k: v for k, v in metrics.items() if k != 'drawdown_series'},
        "trades": trades_df.to_dict('records'),
        "equity_curve": equity_curve.to_dict(),
        "buy_and_hold_equity": buy_hold_equity_series.to_dict(),
        "drawdown_series": metrics['drawdown_series'].to_dict(),
        "historical_data": historical_df.to_dict('index'),
        "debug_info": debug_info
    }