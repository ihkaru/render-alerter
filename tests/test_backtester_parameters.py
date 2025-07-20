# tests/test_backtester_parameters.py

import pytest
import pandas as pd
from backtester import run_backtest
import logging
import copy

# --- Test Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEST - %(levelname)s - %(message)s')

@pytest.fixture
def baseline_config():
    """
    A baseline configuration that is as permissive as possible to establish
    a reliable starting point with the real test data.
    """
    return {
        "strategy_params": {
            "ma_type": "SMA", "short_window": 10, "long_window": 20,
            # --- All Filters Disabled By Default (values are permissive) ---
            "use_price_ma_filter": False,
            "use_rsi_filter": False, "rsi_period": 14, "rsi_buy_threshold": 101,
            "use_price_dist_filter": False, "price_dist_threshold": 0.0,
            "use_cooldown_filter": False, "cooldown_bars": 1,
            "use_hold_duration_filter": False, "min_hold_bars": 1,
            # --- All Exits Disabled By Default ---
            "use_stop_loss": False, "stop_loss_pct": 99.0,
            "use_recovery_exit": False,
            # --- Sizing Parameters ---
            "base_position_pct": 100.0, "leverage_multiplier": 1.0,
        },
        "backtest_settings": {"initial_capital": 10000, "commission_pct": 0.0},
        # Dates that match the downloaded data range
        "strategy_start_timestamp": "2023-05-01T00:00:00Z",
        "strategy_end_timestamp": "2023-08-01T00:00:00Z"
    }

# --- THE MASTER PARAMETER VALIDATION TEST ---

# Note: The 'restrictive_value' dictionaries are calibrated to the real NVDA data.
# Each is designed to specifically change the outcome of the backtest.
@pytest.mark.parametrize("param_to_test, restrictive_value, expectation", [
    # --- Filter Tests (designed to reduce the number of trades) ---
    ("use_rsi_filter", {"use_rsi_filter": True, "rsi_buy_threshold": 45}, "reduce_trades"),
    ("use_price_dist_filter", {"use_price_dist_filter": True, "price_dist_threshold": 1.0}, "reduce_trades"),
    ("use_cooldown_filter", {"use_cooldown_filter": True, "cooldown_bars": 50}, "reduce_trades"),

    # --- Logic Tests (designed to alter trade outcomes) ---
    # ** THE FIX IS HERE **
    # Set hold duration to a value > number of data points to guarantee it forces an end-of-backtest exit.
    ("use_hold_duration_filter", {"use_hold_duration_filter": True, "min_hold_bars": 500}, "check_exit_reason"),
    ("use_stop_loss", {"use_stop_loss": True, "stop_loss_pct": 2.0}, "check_exit_reason"),
    ("leverage_multiplier", {"leverage_multiplier": 5.0}, "check_sizing"),
])
def test_parameter_has_effect_on_backtest(
    mocker,
    baseline_config,
    test_data_provider, # Using the fixture that provides real data
    param_to_test,
    restrictive_value,
    expectation
):
    """
    This is the core validation test. It performs the following steps:
    1. Runs the backtest with a permissive baseline config on REAL data to get a baseline trade count.
    2. Updates the config with a single 'restrictive' parameter setting.
    3. Reruns the backtest.
    4. Asserts that the outcome has changed as expected (e.g., fewer trades, different exit reason).
    This setup ensures that if a parameter's logic is removed from backtester.py, a test will fail.
    """
    # --- ARRANGE ---
    # We don't need to mock the fetcher anymore, as the fixture provides the data.
    
    # --- ACT (Baseline) ---
    baseline_results = run_backtest("NVDA", "stocks", "1h", baseline_config, preloaded_data=test_data_provider)
    assert "trades" in baseline_results and len(baseline_results['trades']) > 0, \
        "Baseline configuration failed to produce any trades with the real test data. The test setup or data is flawed."
    baseline_trade_count = len(baseline_results['trades'])
    baseline_trade = baseline_results['trades'][0]
    logging.info(f"Baseline generated {baseline_trade_count} trades.")

    # --- ARRANGE (Modification) ---
    test_config = copy.deepcopy(baseline_config)
    test_config['strategy_params'].update(restrictive_value)
    logging.info(f"\n--- Running Test: '{param_to_test}' with value: {restrictive_value} ---")

    # --- ACT (Test) ---
    test_results = run_backtest("NVDA", "stocks", "1h", test_config, preloaded_data=test_data_provider)
    test_trade_count = len(test_results.get('trades', []))
    logging.info(f"Test run generated {test_trade_count} trades.")

    # --- ASSERT ---
    if expectation == "reduce_trades":
        assert test_trade_count < baseline_trade_count, \
            f"Parameter '{param_to_test}' was expected to reduce the trade count from {baseline_trade_count}, but resulted in {test_trade_count} trades."

    elif expectation == "check_sizing":
        assert "trades" in test_results and test_trade_count > 0, "Sizing check failed because no trade was executed."
        leveraged_trade = test_results['trades'][0]
        expected_size = baseline_trade['entry_size_usd'] * restrictive_value['leverage_multiplier']
        assert leveraged_trade['entry_size_usd'] == pytest.approx(expected_size), \
            "Leverage multiplier did not scale the position size correctly."

    elif expectation == "check_exit_reason":
        assert "trades" in test_results and test_trade_count > 0, "Exit reason check failed because no trade was executed."
        
        reasons = {trade.get('reason') for trade in test_results['trades']}
        logging.info(f"Observed exit reasons in test run: {reasons}")
        
        if param_to_test == 'use_stop_loss':
            assert "Stop Loss" in reasons, "Expected to find at least one 'Stop Loss' exit, but none were found."
        elif param_to_test == 'use_hold_duration_filter':
            assert "End of Backtest" in reasons, "Expected 'End of Backtest' exit due to long hold duration, but it was not found."