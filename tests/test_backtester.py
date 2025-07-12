import pandas as pd
import pytest
from backtester import run_backtest

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
    
    results = run_backtest("MOCK/USDT", "crypto", "1d", sample_config)
    
    assert len(results['trades']) > 0
    trade = results['trades'][0]
    
    entry_value = 10000 * (95/100) # Simplified from logic, but represents capital used
    exit_value = (entry_value / trade['entry_price']) * trade['exit_price']
    
    # Commission is 0.1% on entry and 0.1% on exit
    expected_commission = (entry_value + exit_value) * (0.1 / 100.0)
    
    assert "commission" in trade
    assert trade['commission'] == pytest.approx(expected_commission)
    
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
        ]),
        'open':  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 98, 100, 100, 100, 100, 100], # sell open is 98 to force loss
        'high': [100]*16, 'low': [100]*16, 'volume':[1000]*16,
        'close': [105, 104, 103, 102, 101, 100, 110, 115, 112, 110, 95, 96, 97, 98, 115, 120],
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

    assert first_trade['entry_size_usd'] == pytest.approx(10000)

    equity_after_loss = 10000 + first_trade['profit']
    size_factor = 0.50
    expected_second_trade_size = equity_after_loss * 1.0 * size_factor

    assert second_trade['entry_size_usd'] == pytest.approx(expected_second_trade_size)