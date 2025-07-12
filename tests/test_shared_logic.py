import pandas as pd
import pytest
from shared_logic import evaluate_buy_filters

# A default set of parameters where all filters are enabled.
# We will toggle them in specific tests.
@pytest.fixture
def all_filters_enabled_params():
    return {
      "use_price_ma_filter": True,
      "use_rsi_filter": True,
      "rsi_buy_threshold": 50,
      "use_price_dist_filter": True,
      "price_dist_threshold": 0.2,
      "use_spread_filter": True,
      "spread_threshold": 0.001,
      "use_volatility_filter": True,
      "atr_min_value": 10,
      "use_volume_filter": True,
      "slope_filter_mode": "Gunakan MA Strategi",
      "strategy_ma_source": "MA Cepat",
      "slope_threshold": 0.1
    }

# A baseline "perfect" candle that should pass all filters.
# We will make it "imperfect" in tests to trigger failures.
@pytest.fixture
def perfect_signal_candle():
    return pd.Series({
        'close': 95,
        'filter_ma': 100,
        'rsi': 45,
        'ma_short': 90,
        'ma_long': 89,
        'atr': 15,
        'volume': 1000,
        'volume_ma': 500,
        'ma_short_slope': 0.5,
        'ma_long_slope': 0.4
    })

def test_all_filters_pass(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test the ideal case where a signal passes all filters. """
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is True
    assert len(failed) == 0

def test_price_vs_ma_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'Price vs Filter MA' """
    perfect_signal_candle['close'] = 105 # Price is now ABOVE the filter_ma
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'Price vs Filter MA' in failed

def test_rsi_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'RSI Buy' """
    perfect_signal_candle['rsi'] = 55 # RSI is now ABOVE the threshold
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'RSI Buy' in failed

def test_price_dist_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'Price Distance' """
    perfect_signal_candle['close'] = 90.1 # Price is too close to ma_short
    # distance = abs(90.1 - 90) / 90 * 100 = 0.11%, which is less than 0.2%
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'Price Distance' in failed

def test_slope_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'Slope Filter' """
    perfect_signal_candle['ma_short_slope'] = 0.05 # Slope is below the 0.1 threshold
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'Slope Filter' in failed
    
def test_volatility_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'Volatility (ATR)' """
    perfect_signal_candle['atr'] = 5 # ATR is below the minimum value of 10
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'Volatility (ATR)' in failed

def test_volume_filter_fail(all_filters_enabled_params, perfect_signal_candle):
    """ Requirement F1.6: Test failure of 'Volume' """
    perfect_signal_candle['volume'] = 400 # Volume is below its moving average
    passed, failed = evaluate_buy_filters(perfect_signal_candle, all_filters_enabled_params)
    assert passed is False
    assert 'Volume' in failed