import json
import pandas as pd
import pytest
from alerter import main_job, send_telegram_alert # <-- MODIFICATION: Added send_telegram_alert
import requests # <-- MODIFICATION: Added for exception testing

# This fixture sets up a temporary directory for our config/state files
@pytest.fixture
def mock_fs(tmp_path):
    # Create a temp directory for our test
    d = tmp_path / "alerter_test"
    d.mkdir()

    # Create dummy config and alert files
    config_data = {"telegram": {"enabled": True}}
    alerts_data = [{
        "id": "BTCUSDT_1h_TEST",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "asset_type": "crypto",
        "params": {
            "short_window": 3,
            "long_window": 5,
            "ma_type": "SMA",
            "use_rsi_filter": False,
            "use_recovery_exit": False, # Disable complex exits for simple tests
            "use_hold_duration_filter": False
        }
    }]

    with open(d / "config.json", 'w') as f:
        json.dump(config_data, f)
    with open(d / "alerts.json", 'w') as f:
        json.dump(alerts_data, f)

    # Yield the path to the test directory
    yield d


def test_alerter_generates_buy_signal(mocker, mock_fs):
    """ Requirement F2.3, F2.4: Test BUY signal generation and state update. """
    # ARRANGE
    # --- MODIFICATION: The lambda now correctly handles keyword arguments. ---
    mocker.patch('alerter.load_json', lambda file, **kwargs: json.load(open(mock_fs / file)))
    mocker.patch('alerter.save_json', lambda file, data: json.dump(data, open(mock_fs / file, 'w')))
    mock_telegram = mocker.patch('alerter.send_telegram_alert')
    mock_fetch = mocker.patch('alerter.fetch_data')

    with open(mock_fs / "state.json", 'w') as f: json.dump({}, f)

    # NEW: Longer DataFrame to allow for 5-period MA calculation and a clear cross
    # The cross happens on the signal candle (iloc[-2])
    data = {
        'timestamp': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00',
                                     '2024-01-01 13:00', '2024-01-01 14:00', # lookbehind candle
                                     '2024-01-01 15:00', # signal candle
                                     '2024-01-01 16:00']),# action candle
        'open':  [100]*7, 'high': [100]*7, 'low': [100]*7, 'volume':[1000]*7,
        'close': [102, 101, 100, 101, 100, 115, 114],
    }
    df = pd.DataFrame(data).set_index('timestamp')
    mock_fetch.return_value = df

    # ACT
    main_job()

    # ASSERT
    mock_telegram.assert_called_once()
    call_args = mock_telegram.call_args[0][0]
    assert "LIVE BUY SIGNAL" in call_args

    with open(mock_fs / "state.json", 'r') as f: state = json.load(f)
    assert state["BTCUSDT_1h_TEST"]["in_position"] is True
    assert state["BTCUSDT_1h_TEST"]["entry_price"] == 115.0 # Close of signal candle

def test_alerter_generates_sell_signal(mocker, mock_fs):
    """ Requirement F2.3, F2.4: Test SELL signal generation and state update. """
    # ARRANGE
    mocker.patch('alerter.load_json', lambda file, **kwargs: json.load(open(mock_fs / file)))
    mocker.patch('alerter.save_json', lambda file, data: json.dump(data, open(mock_fs / file, 'w')))
    mock_telegram = mocker.patch('alerter.send_telegram_alert')
    mock_fetch = mocker.patch('alerter.fetch_data')

    initial_state = { "BTCUSDT_1h_TEST": { "in_position": True, "entry_price": 120.0, "entry_bar_index": 0 } }
    with open(mock_fs / "state.json", 'w') as f: json.dump(initial_state, f)

    # NEW: Longer DataFrame where short_ma_slope turns negative on the signal candle
    data = {
        'timestamp': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00',
                                     '2024-01-01 13:00', '2024-01-01 14:00', # lookbehind candle
                                     '2024-01-01 15:00', # signal candle
                                     '2024-01-01 16:00']),# action candle
        'open':  [100]*7, 'high': [100]*7, 'low': [100]*7, 'volume':[1000]*7,
        'close': [115, 116, 117, 116, 115, 110, 109],
    }
    df = pd.DataFrame(data).set_index('timestamp')
    mock_fetch.return_value = df

    # ACT
    main_job()

    # ASSERT
    mock_telegram.assert_called_once()
    call_args = mock_telegram.call_args[0][0]
    assert "LIVE SELL SIGNAL" in call_args
    assert "P/L:" in call_args

    with open(mock_fs / "state.json", 'r') as f: state = json.load(f)
    assert state["BTCUSDT_1h_TEST"]["in_position"] is False

# --- START OF FIX: The lambda function in all new tests has been corrected. ---
def test_send_telegram_alert_successful(mocker, mock_fs):
    """ Requirement F2.5: Test that a Telegram alert is sent successfully. """
    # ARRANGE
    mocker.patch('alerter.load_json', lambda file, *args, **kwargs: json.load(open(mock_fs / file)))
    mocker.patch('os.getenv', side_effect=lambda key: {'TELEGRAM_TOKEN': 'DUMMY_TOKEN', 'TELEGRAM_CHAT_ID': 'DUMMY_ID'}.get(key))
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.raise_for_status.return_value = None # Simulate a successful request
    
    # ACT
    send_telegram_alert("Test message")

    # ASSERT
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "DUMMY_TOKEN" in args[0] # Check if the URL contains the token
    assert kwargs['json']['chat_id'] == 'DUMMY_ID'
    assert kwargs['json']['text'] == 'Test message'

def test_send_telegram_alert_disabled(mocker, mock_fs):
    """ Requirement F2.5: Test that no alert is sent if Telegram is disabled in the config. """
    # ARRANGE
    disabled_config = {"telegram": {"enabled": False}}
    with open(mock_fs / "config.json", 'w') as f:
        json.dump(disabled_config, f)
    
    mocker.patch('alerter.load_json', lambda file, *args, **kwargs: json.load(open(mock_fs / file)))
    mock_post = mocker.patch('requests.post')

    # ACT
    send_telegram_alert("Test message")

    # ASSERT
    mock_post.assert_not_called()

def test_send_telegram_alert_missing_credentials(mocker, mock_fs, caplog):
    """ Requirement F2.5: Test that an error is logged if credentials are not set. """
    # ARRANGE
    mocker.patch('alerter.load_json', lambda file, *args, **kwargs: json.load(open(mock_fs / file)))
    mocker.patch('os.getenv', return_value=None) # Simulate missing env vars
    mock_post = mocker.patch('requests.post')

    # ACT
    send_telegram_alert("Test message")

    # ASSERT
    mock_post.assert_not_called()
    assert "TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables not set" in caplog.text

def test_send_telegram_alert_api_failure(mocker, mock_fs, caplog):
    """ Requirement F2.5: Test that an error is logged if the API call fails. """
    # ARRANGE
    mocker.patch('alerter.load_json', lambda file, *args, **kwargs: json.load(open(mock_fs / file)))
    mocker.patch('os.getenv', side_effect=lambda key: {'TELEGRAM_TOKEN': 'DUMMY_TOKEN', 'TELEGRAM_CHAT_ID': 'DUMMY_ID'}.get(key))
    mock_post = mocker.patch('requests.post', side_effect=requests.exceptions.RequestException("API Error"))

    # ACT
    send_telegram_alert("Test message")

    # ASSERT
    mock_post.assert_called_once()
    assert "Failed to send Telegram alert" in caplog.text
    assert "API Error" in caplog.text
# --- END OF FIX ---