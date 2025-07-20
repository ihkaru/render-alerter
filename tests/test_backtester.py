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
    
    initial_capital = sample_config['backtest_settings']['initial_capital']
    actual_entry_size = trade['entry_size_usd']
    exit_value = (actual_entry_size / trade['entry_price']) * trade['exit_price']
    
    expected_commission = (actual_entry_size + exit_value) * (sample_config['backtest_settings']['commission_pct'] / 100.0)
    
    assert "commission" in trade
    assert trade['commission'] == pytest.approx(expected_commission, rel=1e-5)

def test_risk_management_size_down_on_loss(mocker, sample_config):
    """ Requirement F1.3: Ensure 'Size Down' rule reduces position size after a loss. """
    # ARRANGE
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09',
            '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13',
            '2024-01-14', '2024-01-15', '2024-01-16', '2024-01-17'
        ]),
        'open':  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 98, 100, 100, 100, 100, 100, 100],
        'high': [100]*17, 'low': [100]*17, 'volume':[1000]*17,
        'close': [105, 104, 103, 102, 101, 100, 110, 115, 112, 110, 95, 96, 97, 98, 115, 120, 121],
    }
    full_df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=full_df)

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

    assert first_trade['entry_size_usd'] == pytest.approx(10000, rel=1e-9)

    equity_after_loss = 10000 + first_trade['profit']
    size_factor = 0.50
    base_position_pct = 1.0
    expected_second_trade_size = equity_after_loss * base_position_pct * size_factor

    assert second_trade['entry_size_usd'] == pytest.approx(expected_second_trade_size, rel=1e-9)

def test_end_of_period_position_closure(mocker, sample_config):
    """ F1.7: Test if an open position is closed on the last candle. """
    mock_df = create_mock_dataframe()
    mock_df.loc['2024-01-06', 'close'] = 100
    mock_df.loc['2024-01-07', 'close'] = 115
    mock_df.loc['2024-01-08', 'close'] = 120
    mock_df.loc['2024-01-09', 'close'] = 125
    mock_df.loc['2024-01-10', 'close'] = 130
    mock_df.loc['2024-01-11', 'close'] = 135
    mock_df.loc['2024-01-12', 'close'] = 140

    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    sample_config['strategy_params']['use_rsi_filter'] = False
    sample_config['strategy_params']['use_stop_loss'] = False

    debug_indicators = calculate_indicators(mock_df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators.iloc[-5:], "End of Period Closure Data")

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "error" not in results, f"Backtest failed with error: {results.get('error')}"
    assert len(results['trades']) == 1, "A trade should have been opened and force-closed"
    trade = results['trades'][0]

    last_candle_close = mock_df['close'].iloc[-1]
    last_candle_ts = mock_df.index[-1]

    assert trade['reason'] == "End of Backtest"
    assert trade['exit_price'] == last_candle_close
    assert trade['exit_ts'] == last_candle_ts
    assert 'equity_curve' in results
    final_equity = list(results['equity_curve'].values())[-1]
    assert final_equity > sample_config['backtest_settings']['initial_capital']


def test_stop_loss_trigger_on_low(mocker, sample_config):
    """ Requirement F1.3: Test if stop loss triggers on the candle's LOW price. """
    mock_df = create_mock_dataframe()
    mock_df.loc['2024-01-06', 'close'] = 100
    mock_df.loc['2024-01-07', 'close'] = 115
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)
    
    sample_config['strategy_params']['use_rsi_filter'] = False
    sample_config['strategy_params']['use_stop_loss'] = True
    sample_config['strategy_params']['stop_loss_pct'] = 10.0
    
    expected_sl_price = 115 * 0.9
    
    mock_df.loc['2024-01-09', 'open'] = 114
    mock_df.loc['2024-01-09', 'low'] = 103
    mock_df.loc['2024-01-09', 'close'] = 113

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "error" not in results
    assert len(results['trades']) == 1, "A trade should have been stopped out"
    trade = results['trades'][0]

    assert trade['reason'] == "Stop Loss"
    assert trade['exit_price'] == expected_sl_price
    assert trade['exit_ts'] == pd.Timestamp('2024-01-09 00:00:00+0000', tz='UTC')


def test_cooldown_filter_prevents_buy(mocker, sample_config):
    """ F1.3: Test 'Cooldown Period' filter rejects a new buy signal. """
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08',
            '2024-01-09', '2024-01-10', '2024-01-11',
            '2024-01-12', '2024-01-13', '2024-01-14',
        ]),
        'open':  [100]*14, 'high': [100]*14, 'low': [100]*14, 'volume':[1000]*14,
        'close': [100, 100, 100, 100, 100, 100, 115, 120, 122, 110, 108, 107, 125, 130],
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False,
        'use_stop_loss': False,
        'use_cooldown_filter': True,
        'cooldown_bars': 3
    })

    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Final Cooldown Filter Data")

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "trades" in results, f"Backtest failed or produced no trades. Debug info: {results.get('debug_info')}"
    assert len(results['trades']) == 1, "Only the first trade should execute due to the cooldown filter"

    trade = results['trades'][0]
    assert trade['entry_ts'] == pd.Timestamp('2024-01-08 00:00:00+0000', tz='UTC')
    assert trade['exit_ts'] == pd.Timestamp('2024-01-11 00:00:00+0000', tz='UTC')



def test_risk_management_fake_loss_rule(mocker, sample_config):
    """ F1.3: Test 'Fake Loss' rule reduces equity used for sizing. """
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09',
            '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13',
            '2024-01-14', '2024-01-15', '2024-01-16',
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

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

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
    """ F1.3: Test 'Recovery Exit' triggers correctly. """
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-01-05', '2024-01-06', '2024-01-07',
            '2024-01-08', '2024-01-09', '2024-01-10',
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
        'use_hold_duration_filter': True,
        'min_hold_bars': 100,
    })
    sample_config['backtest_settings']['commission_pct'] = 0.0
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Final Isolated Recovery Exit Data")

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    logging.info(f"Test Results: {results}")
    assert "trades" in results and len(results['trades']) == 1, f"Expected 1 trade, but got results: {results}"
    trade = results['trades'][0]

    debug_df(pd.DataFrame(results['trades']), "Recovery Exit Trade")

    assert trade['reason'] == "Recovery Exit"
    assert trade['entry_price'] == 100.0
    assert trade['exit_ts'] == pd.Timestamp('2024-01-10 00:00:00+0000', tz='UTC')


def test_indicator_warmup_delays_start(mocker, sample_config):
    """ F1.4: Test that backtest start is aligned with valid indicators. """
    mock_df = create_mock_dataframe()
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)

    sample_config['strategy_start_timestamp'] = "2024-01-03T00:00:00Z"
    sample_config['strategy_params']['long_window'] = 5

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "historical_data" in results, "Backtest should return historical_data even if no trades."
    
    historical_data_df = pd.DataFrame.from_dict(results['historical_data'], orient='index')
    historical_data_df.index = pd.to_datetime(historical_data_df.index)

    actual_start_ts = historical_data_df.index.min()
    
    expected_actual_start_ts = pd.Timestamp("2024-01-05T00:00:00Z")

    logging.info(f"User requested start: {sample_config['strategy_start_timestamp']}")
    logging.info(f"Expected actual start: {expected_actual_start_ts}")
    logging.info(f"Actual first data timestamp: {actual_start_ts}")

    assert actual_start_ts == expected_actual_start_ts

def test_min_hold_duration_prevents_early_exit(mocker, sample_config):
    """ F1.3/F2.3: Test 'min_hold_bars' prevents a premature sell signal. """
    data = {
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
                                     '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08']),
        'open':  [100]*8, 'high': [100]*8, 'low': [100]*8, 'volume':[1000]*8,
        'close': [102, 101, 100, 99, 98, 115, 100, 101],
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False, 'use_stop_loss': False, 'use_recovery_exit': False,
        'use_hold_duration_filter': True,
        'min_hold_bars': 3
    })
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "trades" in results and len(results['trades']) == 1, f"Test failed. Results: {results}"
    trade = results['trades'][0]
    
    debug_df(pd.DataFrame(results['trades']), "Min Hold Duration Test")
    
    assert trade['reason'] == "End of Backtest", "Sell signal should have been ignored due to min_hold_bars"
    assert trade['exit_ts'] == pd.Timestamp('2024-01-08 00:00:00+0000', tz='UTC')


def test_sell_signal_filtered_by_rsi(mocker, sample_config):
    """ F1.3/F2.3: Test Sell signal (MA Slope AND RSI Filter) triggers an exit. """
    # ARRANGE
    # This data is crafted to create a Golden Cross, a price run-up, and then a
    # gradual decline to make the MA slope negative while keeping RSI high.
    data = {
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', # Lookbehind for Buy
            '2024-01-07', # Signal for Buy (Golden Cross)
            '2024-01-08', # Action for Buy, price run-up
            '2024-01-09', # Price peaks, RSI is very high
            # --- MODIFICATION: Gradual decline to keep RSI high but make slope negative ---
            '2024-01-10', # Signal for Sell (Slope turns negative, RSI still high)
            '2024-01-11', # Action for Sell
        ]),
        'open':  [100]*11, 'high': [100]*11, 'low': [100]*11, 'volume':[1000]*11,
        # Modified: Ensuring negative slope by 2024-01-10
        'close': [100, 100, 100, 100, 100, 105, 115, 125, 123, 114, 108], # <-- Fixed data
    }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': True,
        'rsi_buy_threshold': 101,     # Set high to allow entry where RSI is 100
        'rsi_sell_threshold': 40,     # Lowered to match actual RSI when slope turns negative
        'rsi_period': 3,
        'short_window': 3,
        'long_window': 5,
        'use_stop_loss': False, 'use_recovery_exit': False, 'use_hold_duration_filter': False
    })
    sample_config['strategy_start_timestamp'] = "2024-01-01T00:00:00Z"

    debug_indicators = calculate_indicators(df.copy(), sample_config['strategy_params'])
    debug_df(debug_indicators, "Sell Signal with RSI Filter Data")

    # ACT
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    # ASSERT
    assert "trades" in results and len(results['trades']) == 1, f"Test failed. Results: {results}"
    trade = results['trades'][0]
    debug_df(pd.DataFrame(results['trades']), "Sell Signal with RSI Filter Test")

    # The buy signal is on 2024-01-07, executes on 2024-01-08.
            # The sell signal should trigger on 2024-01-10 (MA slope negative, RSI > 40).
    # The exit happens on the next candle's open, 2024-01-11.
    assert trade['reason'] == "Sell Signal"
    assert trade['exit_ts'] == pd.Timestamp('2024-01-11 00:00:00+00:00', tz='UTC')
    
def test_leverage_increases_position_size(mocker, sample_config):
    """ Requirement F4.1: Test leverage multiplier correctly scales entry_size_usd. """
    mock_df = create_mock_dataframe()
    mock_df.loc['2024-01-07', 'close'] = 115
    mocker.patch('backtester.fetch_full_historical_data', return_value=mock_df)

    sample_config['strategy_params'].update({
        'use_rsi_filter': False,
        'base_position_pct': 95.0,
        'leverage_multiplier': 3.0
    })
    initial_capital = sample_config['backtest_settings']['initial_capital']
    
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    assert "trades" in results and len(results['trades']) > 0
    trade = results['trades'][0]
    
    expected_size = initial_capital * 0.95 * 3.0
    
    logging.info(f"Expected Size: {expected_size}, Actual Size: {trade['entry_size_usd']}")
    assert trade['entry_size_usd'] == pytest.approx(expected_size)


def test_engine_handles_flat_line_data(mocker, sample_config):
    """ NFR Test: Ensure no trades or errors on flat price data. """
    data = {'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
                                         '2024-01-05', '2024-01-06', '2024-01-07']),
            'open': [100]*7, 'high': [100]*7, 'low': [100]*7, 'volume':[1000]*7, 'close': [100]*7 }
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)
    
    assert results['message'] == "No trades were executed."
    assert results['debug_info']['crossover_events_found'] == 0


def test_engine_handles_insufficient_data(mocker, sample_config):
    """ F1.4 Test: Ensure graceful error on data shorter than warmup period. """
    data = {'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'open': [100]*3, 'high': [100]*3, 'low': [100]*3, 'volume':[1000]*3, 'close': [100]*3}
    df = pd.DataFrame(data).set_index('timestamp').tz_localize('UTC')
    mocker.patch('backtester.fetch_full_historical_data', return_value=df)
    
    sample_config['strategy_params']['long_window'] = 5

    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)

    logging.info(f"Results for insufficient data: {results}")
    assert "error" in results
    assert "Not enough data" in results['error']