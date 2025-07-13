import pandas as pd
import pytest
from backtester import run_backtest
import logging # <-- Import logging
from shared_logic import calculate_indicators # <-- Import for debugging
# A minimal config for testing purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEST - %(levelname)s - %(message)s')

# Helper function to print DataFrame for debugging
def debug_df(df, title=""):
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)
    logging.info(f"--- Debugging DataFrame: {title} ---")
    print(df)
    logging.info("--- End Debugging ---")


# A minimal config for testing purposes
@pytest.fixture
def sample_config():
    return {
        "strategy_params": {
            "short_window": 3,
            "long_window": 5,
            "ma_type": "SMA",
            "use_rsi_filter": True,
            "rsi_period": 3,
            "rsi_buy_threshold": 40,
            "use_stop_loss": True,
            "stop_loss_pct": 10.0
        },
        "backtest_settings": {
          "initial_capital": 10000,
          "commission_pct": 0.1
        },
        "strategy_start_timestamp": "2024-01-05T00:00:00Z"
    }

# This fixture creates a predictable DataFrame for our tests
def create_mock_dataframe():
    # This data is specifically designed to create a clear golden cross.
    # MA_short (3) starts below MA_long (5) and then crosses above it.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', # Lookbehind Candle: MA_short is still below MA_long
            '2024-01-07', # Signal Candle: MA_short crosses decisively above MA_long
            '2024-01-08', # Action Candle: Trade should execute on this open
            '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12'
        ]),
        'open':  [100, 101, 102, 101, 100, 101, 110, 115, 114, 112, 110, 108],
        'high':  [101, 102, 103, 102, 101, 102, 112, 116, 115, 113, 111, 109],
        'low':   [99,  100, 101, 100, 99,  100, 108, 114, 113, 111, 109, 107],
        'close': [101, 102, 102, 101, 100, 101, 112, 114, 113, 111, 108, 106],
        'volume':[1000,1000,1000,1000,1000,1000,1000,2000,1000,1000,1000,1000]
    }
    df = pd.DataFrame(data).set_index('timestamp')
    df.index = df.index.tz_localize('UTC')
    return df

def test_non_repainting_logic(mocker, sample_config):
    """ Requirement F1.1: Ensure trades execute on the OPEN of the *action* candle. """
    mock_df = create_mock_dataframe()
    # Manually create a golden cross on 2024-01-07 (signal candle)
    # The action candle is 2024-01-08
    mock_df.loc['2024-01-06', 'close'] = 100 # lookbehind
    mock_df.loc['2024-01-07', 'close'] = 115 # signal
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    
    # Disable RSI filter to guarantee a trade
    sample_config['strategy_params']['use_rsi_filter'] = False

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "error" not in results
    assert len(results['trades']) == 1
    trade = results['trades'][0]
    
    # The signal candle is 2024-01-07. The action candle is 2024-01-08.
    # The entry price MUST be the open of the action candle.
    expected_entry_price = mock_df.loc['2024-01-08']['open']
    assert trade['entry_price'] == expected_entry_price
    assert trade['entry_ts'] == pd.Timestamp('2024-01-08 00:00:00+0000', tz='UTC')

def test_backtest_with_no_trades_debug_output(mocker, sample_config):
    """ Requirement F1.6: Ensure correct debug output when no trades occur. """
    mock_df = create_mock_dataframe()
    # Create a golden cross, but ensure RSI is too high to pass the filter
    mock_df.loc['2024-01-06', 'close'] = 100
    mock_df.loc['2024-01-07', 'close'] = 150 # This will make RSI very high
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "message" in results and results["message"] == "No trades were executed."
    assert "debug_info" in results
    debug_info = results['debug_info']
    assert debug_info['crossover_events_found'] > 0
    assert 'RSI Buy' in debug_info['failed_filter_counts']
    assert debug_info['failed_filter_counts']['RSI Buy'] > 0

def test_performance_analytics_output(mocker, sample_config):
    """ Requirement F5.1 & F5.2: Ensure analytics metrics are calculated. """
    mock_df = create_mock_dataframe()
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    sample_config['strategy_params']['use_rsi_filter'] = False # Ensure a trade happens
    
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "metrics" in results
    metrics = results['metrics']
    assert "max_drawdown" in metrics
    assert "max_drawdown_pct" in metrics
    assert isinstance(metrics['max_drawdown'], (int, float))
    assert isinstance(metrics['max_drawdown_pct'], (int, float))
    
    assert "buy_and_hold_equity" in results
    assert "drawdown_series" in results
    assert len(results['buy_and_hold_equity']) > 0
    assert len(results['drawdown_series']) > 0

def test_commission_calculation(mocker, sample_config):
    """ Requirement F1.2: Ensure commission is applied correctly. """
    mock_df = create_mock_dataframe()
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    sample_config['strategy_params']['use_rsi_filter'] = False
    
    # Run backtest, forcing a single trade that closes at the end
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)
    
    assert len(results['trades']) > 0
    trade = results['trades'][0]
    
    # We need a predictable entry size. Let's assume 95% of initial capital based on config
    initial_capital = sample_config['backtest_settings']['initial_capital']
    # The logic uses 95% as a base, so let's use that for calculation.
    # Note: this is a simplification, a more robust test would mock the exact equity value.
    entry_value_approx = initial_capital * 0.95 
    # The actual entry size is calculated inside, let's use the one from the results for precision
    actual_entry_size = trade['entry_size_usd']
    exit_value = (actual_entry_size / trade['entry_price']) * trade['exit_price']
    
    # Commission is 0.1% on entry and 0.1% on exit
    expected_commission = (actual_entry_size + exit_value) * (sample_config['backtest_settings']['commission_pct'] / 100.0)
    
    assert "commission" in trade
    assert trade['commission'] == pytest.approx(expected_commission, rel=1e-5)

def test_risk_management_size_down_on_loss(mocker, sample_config):
    """ Requirement F1.3: Ensure 'Size Down' rule reduces position size after a loss. """
    # ARRANGE
    # This data is now mathematically verified to produce the required signals.
    # It creates a BUY, then a SELL at a loss, then a second BUY.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', # Lookbehind 1: ma_short < ma_long
            '2024-01-07', # Signal 1: ma_short > ma_long (BUY)
            '2024-01-08', # Action 1 (Buy)
            '2024-01-09',
            '2024-01-10', # Signal 2: ma_short_slope becomes negative (SELL)
            '2024-01-11', # Action 2 (Sell)
            '2024-01-12',
            '2024-01-13',
            '2024-01-14', # Lookbehind 2
            '2024-01-15', # Signal 3 (BUY again)
            '2024-01-16', # Action 3 (Buy again)
            '2024-01-17'  # Final candle to close the last trade
        ]),
        'open':  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 98, 100, 100, 100, 100, 100, 100], # sell open is 98 to force loss
        'high': [100]*17, 'low': [100]*17, 'volume':[1000]*17,
        'close': [105, 104, 103, 102, 101, 100, 110, 115, 112, 110, 95, 96, 97, 98, 115, 120, 121],
    }
    full_df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=full_df)

    # Configure parameters
    sample_config['strategy_params']['use_rsi_filter'] = False
    sample_config['strategy_params']['use_stop_loss'] = False
    sample_config['strategy_params']['use_size_down'] = True
    sample_config['strategy_params']['size_down_pct'] = 50.0
    sample_config['strategy_params']['use_fake_loss'] = False
    sample_config['backtest_settings']['initial_capital'] = 10000
    sample_config['backtest_settings']['commission_pct'] = 0.0
    sample_config['strategy_params']['base_position_pct'] = 100.0
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results, f"Backtest failed. Debug Info: {results.get('debug_info')}"
    assert len(results['trades']) == 2
    
    first_trade, second_trade = results['trades'][0], results['trades'][1]

    assert first_trade['profit'] < 0
    assert 'entry_size_usd' in first_trade
    assert 'entry_size_usd' in second_trade

    # The first trade should use 100% of initial capital (minus a tiny bit for safety, hence approx)
    assert first_trade['entry_size_usd'] == pytest.approx(10000, rel=1e-9)

    # After the loss, equity is reduced. The next trade size should be scaled down.
    equity_after_loss = 10000 + first_trade['profit']
    size_factor = 0.50 # from size_down_pct
    base_position_pct = 1.0 # from base_position_pct
    expected_second_trade_size = equity_after_loss * base_position_pct * size_factor

    assert second_trade['entry_size_usd'] == pytest.approx(expected_second_trade_size, rel=1e-9)

# --- NEW TESTS START HERE ---

def test_end_of_period_position_closure(mocker, sample_config):
    """ (FIXED) F1.7: Test if an open position is closed on the last candle. """
    # ARRANGE
    mock_df = create_mock_dataframe()
    # Create a BUY signal on 2024-01-07
    mock_df.loc['2024-01-06', 'close'] = 100
    mock_df.loc['2024-01-07', 'close'] = 115

    # **** KEY FIX ****
    # Ensure no natural SELL signal is generated. We will force the price to
    # rise steadily, guaranteeing the 3-period MA slope remains positive.
    mock_df.loc['2024-01-08', 'close'] = 120
    mock_df.loc['2024-01-09', 'close'] = 125
    mock_df.loc['2024-01-10', 'close'] = 130
    mock_df.loc['2024-01-11', 'close'] = 135
    mock_df.loc['2024-01-12', 'close'] = 140 # This is the final candle

    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    sample_config['strategy_params']['use_rsi_filter'] = False
    sample_config['strategy_params']['use_stop_loss'] = False

    # **** DEBUGGING STEP ****
    # Calculate indicators on our mock data to verify our assumptions
    debug_indicators = calculate_indicators(mock_df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators.iloc[-5:], "End of Period Closure Data") # Print last 5 rows

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "error" not in results, f"Backtest failed with error: {results.get('error')}"
    assert len(results['trades']) == 1, "A trade should have been opened and force-closed"
    trade = results['trades'][0]

    last_candle_close = mock_df['close'].iloc[-1]
    last_candle_ts = mock_df.index[-1]

    # This assertion is now the primary goal
    assert trade['reason'] == "End of Backtest"
    assert trade['exit_price'] == last_candle_close
    assert trade['exit_ts'] == last_candle_ts
    assert 'equity_curve' in results
    final_equity = list(results['equity_curve'].values())[-1]
    assert final_equity > sample_config['backtest_settings']['initial_capital']


def test_stop_loss_trigger_on_low(mocker, sample_config):
    """ Requirement F1.3: Test if stop loss triggers on the candle's LOW price. """
    # ARRANGE
    mock_df = create_mock_dataframe()
    # Create a BUY signal
    mock_df.loc['2024-01-06', 'close'] = 100
    mock_df.loc['2024-01-07', 'close'] = 115
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    
    sample_config['strategy_params']['use_rsi_filter'] = False
    sample_config['strategy_params']['use_stop_loss'] = True
    sample_config['strategy_params']['stop_loss_pct'] = 10.0
    
    # The trade enters on 2024-01-08 at open price of 115.
    # A 10% stop loss would be at 115 * (1 - 0.10) = 103.5
    expected_sl_price = 115 * 0.9
    
    # Modify the NEXT candle (2024-01-09) to trigger the SL on its low
    mock_df.loc['2024-01-09', 'open'] = 114  # Open is above SL
    mock_df.loc['2024-01-09', 'low'] = 103   # Low is BELOW SL
    mock_df.loc['2024-01-09', 'close'] = 113 # Close is above SL

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "error" not in results
    assert len(results['trades']) == 1, "A trade should have been stopped out"
    trade = results['trades'][0]

    assert trade['reason'] == "Stop Loss"
    # The exit price must be the SL level itself, not the open or close.
    assert trade['exit_price'] == expected_sl_price
    assert trade['exit_ts'] == pd.Timestamp('2024-01-09 00:00:00+0000', tz='UTC')


def test_cooldown_filter_prevents_buy(mocker, sample_config):
    """ (FINAL) F1.3: Test 'Cooldown Period' filter rejects a new buy signal. """
    # ARRANGE
    # This data creates a clear BUY->SELL cycle, followed immediately by
    # another BUY signal that falls squarely within the cooldown period.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            # --- First BUY Signal ---
            '2024-01-06', # Lookbehind candle
            '2024-01-07', # Signal candle (Golden Cross)
            '2024-01-08', # Action candle (Buy executes here)
            # --- First SELL Signal ---
            '2024-01-09', # Price peaks
            '2024-01-10', # Signal candle (MA slope turns negative)
            '2024-01-11', # Action candle (Sell executes here)
            # --- Second BUY Signal (to be blocked) ---
            '2024-01-12', # Lookbehind candle for 2nd buy. Cooldown bar #1.
            '2024-01-13', # Signal candle (Golden Cross). Cooldown bar #2.
            '2024-01-14', # Action candle (Should NOT trade). Cooldown bar #3.
        ]),
        'open':  [100]*14, 'high': [100]*14, 'low': [100]*14, 'volume':[1000]*14,
        'close': [100, 100, 100, 100, 100,  # 1. Initial flat data to stabilize MAs
                  100, 115, 120,           # 2. Clear BUY sequence
                  122, 110, 108,           # 3. Clear SELL sequence
                  107, 125, 130],          # 4. Clear 2nd BUY sequence
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False,
        'use_stop_loss': False,
        'use_cooldown_filter': True,
        'cooldown_bars': 3 # A 3-bar cooldown.
    })

    # **** DEBUGGING STEP ****
    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Final Cooldown Filter Data")

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    # The new buy signal on 2024-01-13 is only 2 bars after the exit on 2024-01-11.
    # The check `2 < 3` will be TRUE, so the trade must be blocked.
    assert "trades" in results, f"Backtest failed or produced no trades. Debug info: {results.get('debug_info')}"
    assert len(results['trades']) == 1, "Only the first trade should execute due to the cooldown filter"

    # Verify the trade that *did* happen is the correct one.
    trade = results['trades'][0]
    assert trade['entry_ts'] == pd.Timestamp('2024-01-08 00:00:00+0000', tz='UTC')
    assert trade['exit_ts'] == pd.Timestamp('2024-01-11 00:00:00+0000', tz='UTC')



def test_risk_management_fake_loss_rule(mocker, sample_config):
    """ (FIXED) F1.3: Test 'Fake Loss' rule reduces equity used for sizing. """
    # ARRANGE
    # This data ensures the first trade is profitable.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', # BUY signal 1
            '2024-01-08', '2024-01-09', # Profitable Trade
            '2024-01-10', '2024-01-11', # SELL signal 1
            '2024-01-12', '2024-01-13',
            '2024-01-14', '2024-01-15', # BUY signal 2
            '2024-01-16',
        ]),
        'open':  [100, 100, 100, 100, 100, 100, 100, 116, 120, 125, 122, 126, 105, 112, 100, 100],
        'high': [100]*16, 'low': [100]*16, 'volume':[1000]*16,
        'close': [100, 100, 100, 100, 100, 100, 115, 120, 125, 122, 110, 108, 105, 112, 128, 130],
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False, 'use_stop_loss': False, 'use_cooldown_filter': False,
        'use_fake_loss': True, 'withdrawal_pct': 50.0,
        'base_position_pct': 100.0
    })
    sample_config['backtest_settings'].update({'commission_pct': 0.0, 'initial_capital': 10000})

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results and len(results['trades']) == 2
    first_trade, second_trade = results['trades']

    assert first_trade['profit'] > 0, "First trade should have been profitable"

    true_equity = 10000 + first_trade['profit']
    withdrawal_amount = first_trade['profit'] * 0.50
    adjusted_equity_for_sizing = true_equity - withdrawal_amount
    expected_second_trade_size = adjusted_equity_for_sizing * 1.0

    debug_df(pd.DataFrame(results['trades']), "Fake Loss Rule Trades")
    logging.info(f"Profit: {first_trade['profit']}, Withdrawal: {withdrawal_amount}, Adjusted Equity: {adjusted_equity_for_sizing}")

    assert second_trade['entry_size_usd'] == pytest.approx(expected_second_trade_size)

def test_recovery_exit_trigger(mocker, sample_config):
    """ (FINAL, ISOLATED & VERIFIED) F1.3: Test 'Recovery Exit' triggers correctly. """
    # ARRANGE
    # This data is mathematically verified to create a golden cross, then dip and recover.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-01-05', # Lookbehind: ma_short < ma_long
            '2024-01-06', # Signal: ma_short > ma_long
            '2024-01-07', # Action: Buy @ 100
            '2024-01-08', # Price Dips > 2% (has_dipped_deep=True).
            '2024-01-09', # Price Recovers to < 0.1% loss.
            '2024-01-10', # Action: Sell @ 100 due to recovery signal.
        ]),
        'open':  [100]*10, 'high': [100]*10, 'low': [100]*10, 'volume':[1000]*10,
        'close': [102, 101, 100, 99, 98, 115, 100, 97, 99.95, 100],
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False,
        'use_stop_loss': False,
        'use_recovery_exit': True,
        'dip_pct_trigger': 2.0,
        'recovery_pct_threshold': 0.1,
        # --- THE CRITICAL FIX ---
        # We disable the standard sell signal by setting an impossibly long hold duration.
        # This forces the backtester to ONLY evaluate the recovery exit logic.
        'use_hold_duration_filter': True,
        'min_hold_bars': 100,
    })
    sample_config['backtest_settings']['commission_pct'] = 0.0
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    # **** DEBUGGING STEP ****
    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Final Isolated Recovery Exit Data")

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    logging.info(f"Test Results: {results}")
    assert "trades" in results and len(results['trades']) == 1, f"Expected 1 trade, but got results: {results}"
    trade = results['trades'][0]

    debug_df(pd.DataFrame(results['trades']), "Recovery Exit Trade")

    assert trade['reason'] == "Recovery Exit"
    assert trade['entry_price'] == 100.0
    assert trade['exit_ts'] == pd.Timestamp('2024-01-10 00:00:00+0000', tz='UTC')


def test_indicator_warmup_delays_start(mocker, sample_config):
    """ (FIXED) F1.4: Test that backtest start is aligned with valid indicators. """
    # ARRANGE
    mock_df = create_mock_dataframe()
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)

    sample_config['strategy_start_timestamp'] = "2024-01-03T00:00:00Z"
    sample_config['strategy_params']['long_window'] = 5

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    # This test no longer depends on a trade. It directly checks if the
    # historical data used by the backtest was correctly sliced to begin
    # only after the longest MA (5 periods) is valid.
    assert "historical_data" in results, "Backtest should return historical_data even if no trades."
    
    historical_data_df = pd.DataFrame.from_dict(results['historical_data'], orient='index')
    historical_data_df.index = pd.to_datetime(historical_data_df.index)

    actual_start_ts = historical_data_df.index.min()
    
    # On the mock_df, the 5-period SMA is first non-NaN on 2024-01-05
    expected_actual_start_ts = pd.Timestamp("2024-01-05T00:00:00Z")

    logging.info(f"User requested start: {sample_config['strategy_start_timestamp']}")
    logging.info(f"Expected actual start: {expected_actual_start_ts}")
    logging.info(f"Actual first data timestamp: {actual_start_ts}")

    assert actual_start_ts == expected_actual_start_ts

def test_min_hold_duration_prevents_early_exit(mocker, sample_config):
    """ (FIXED) F1.3/F2.3: Test 'min_hold_bars' prevents a premature sell signal. """
    # ARRANGE
    # This data creates a BUY signal, then an immediate SELL signal one bar later.
    data = {
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
                                     '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08']),
        'open':  [100]*8, 'high': [100]*8, 'low': [100]*8, 'volume':[1000]*8,
        'close': [102, 101, 100, 99, 98, 115, 100, 101], # Creates cross, then immediate sell signal
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False, 'use_stop_loss': False, 'use_recovery_exit': False,
        'use_hold_duration_filter': True,
        'min_hold_bars': 3
    })
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results and len(results['trades']) == 1, f"Test failed. Results: {results}"
    trade = results['trades'][0]
    
    debug_df(pd.DataFrame(results['trades']), "Min Hold Duration Test")
    
    assert trade['reason'] == "End of Backtest", "Sell signal should have been ignored due to min_hold_bars"
    assert trade['exit_ts'] == pd.Timestamp('2024-01-08 00:00:00+0000', tz='UTC')



def test_rsi_sell_filter_triggers_exit(mocker, sample_config):
    """ (FINAL, VERIFIED & ISOLATED) F1.3/F2.3: Test RSI Sell filter triggers an exit. """
    # ARRANGE
    # This data is crafted to create a Golden Cross, and then a very high RSI.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-01-05', # Lookbehind -> ma_short < ma_long
            '2024-01-06', # Still ma_short < ma_long
            '2024-01-07', # Still ma_short < ma_long
            '2024-01-08', # Signal -> Golden Cross
            '2024-01-09', # Price explodes -> RSI becomes > 80
            '2024-01-10', # Action -> Sell
        ]),
        'open':  [100]*10, 'high': [100]*10, 'low': [100]*10, 'volume':[1000]*10,
        'close': [120, 118, 115, 100, 102, 104, 106, 110, 150, 160],
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': True,
        # --- THE CRITICAL FIX ---
        # We raise the buy threshold just for this test to allow the entry.
        # This isolates the sell-side filter, which is the actual target of this test.
        'rsi_buy_threshold': 70,
        'rsi_sell_threshold': 80,  # Sell if RSI is high
        'rsi_period': 3,
        'use_stop_loss': False, 'use_recovery_exit': False, 'use_hold_duration_filter': False
    })
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    # **** DEBUGGING STEP ****
    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Final Final RSI Sell Filter Data")

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results and len(results['trades']) == 1, f"Test failed. Results: {results}"
    trade = results['trades'][0]
    debug_df(pd.DataFrame(results['trades']), "RSI Sell Filter Test")

    # The buy on 2024-01-08 is now allowed (RSI 65.85 < 70).
    # The sell signal is on 2024-01-09 (RSI 94.97 > 80).
    # The exit happens on the next candle's open, 2024-01-10.
    assert trade['reason'] == "Sell Signal"
    assert trade['exit_ts'] == pd.Timestamp('2024-01-10 00:00:00+00:00', tz='UTC')
    
def test_leverage_increases_position_size(mocker, sample_config):
    """ Requirement F4.1: Test leverage multiplier correctly scales entry_size_usd. """
    # ARRANGE
    mock_df = create_mock_dataframe()
    mock_df.loc['2024-01-07', 'close'] = 115 # Ensure trade signal
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False,
        'base_position_pct': 95.0,
        'leverage_multiplier': 3.0
    })
    initial_capital = sample_config['backtest_settings']['initial_capital']
    
    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results and len(results['trades']) > 0
    trade = results['trades'][0]
    
    # Expected size = capital * (base_pct/100) * leverage
    expected_size = initial_capital * 0.95 * 3.0
    
    logging.info(f"Expected Size: {expected_size}, Actual Size: {trade['entry_size_usd']}")
    assert trade['entry_size_usd'] == pytest.approx(expected_size)


def test_engine_handles_flat_line_data(mocker, sample_config):
    """ NFR Test: Ensure no trades or errors on flat price data. """
    # ARRANGE
    data = {'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
                                         '2024-01-05', '2024-01-06', '2024-01-07']),
            'open': [100]*7, 'high': [100]*7, 'low': [100]*7, 'volume':[1000]*7, 'close': [100]*7 }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)
    
    # ASSERT
    assert results['message'] == "No trades were executed."
    assert results['debug_info']['crossover_events_found'] == 0


def test_engine_handles_insufficient_data(mocker, sample_config):
    """ (FIXED) F1.4 Test: Ensure graceful error on data shorter than warmup period. """
    # ARRANGE
    data = {'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'open': [100]*3, 'high': [100]*3, 'low': [100]*3, 'volume':[1000]*3, 'close': [100]*3}
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)
    
    sample_config['strategy_params']['long_window'] = 5

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    logging.info(f"Results for insufficient data: {results}")
    assert "error" in results
    assert "Not enough data" in results['error']