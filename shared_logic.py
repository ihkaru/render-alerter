import logging
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import ccxt
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

def load_json(filepath: str, default_value=None):
    """Loads a JSON file with a fallback to a default value."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_value if default_value is not None else {}

def fetch_full_historical_data(asset_type: str, symbol: str, timeframe: str, start_date_str: str, warmup_timedelta: pd.Timedelta) -> Optional[pd.DataFrame]:
    # This function is correct, no changes needed
    logging.info(f"Performing deep historical fetch for {symbol}...")
    user_start_date = pd.to_datetime(start_date_str)
    fetch_start_date = user_start_date - warmup_timedelta
    try:
        if asset_type == 'stocks':
            df = yf.download(tickers=symbol, start=fetch_start_date, end=datetime.now(), interval=timeframe.replace('d', 'd').replace('h', 'h'), progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        elif asset_type == 'crypto':
            exchange = getattr(ccxt, 'binance')()
            all_ohlcv = []
            since = int(fetch_start_date.timestamp() * 1000)
            limit = 1000
            retries = 3
            while since < int(datetime.now().timestamp() * 1000):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                    if not ohlcv: break
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    time.sleep(exchange.rateLimit / 1000)
                except Exception as e:
                    logging.warning(f"API fetch failed: {e}. Retries left: {retries}")
                    retries -= 1
                    if retries <= 0: return None 
                    time.sleep(5) 
            if not all_ohlcv: return None
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.drop_duplicates(subset='timestamp', inplace=True)
            df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            df.sort_index(inplace=True)
        else: return None
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')
        return df.astype(float, errors='ignore')
    except Exception as e:
        logging.error(f"Failed during deep historical fetch for {symbol}: {e}", exc_info=True)
        return None

def fetch_data(asset_type: str, symbol: str, timeframe: str, params: Dict[str, Any], historical_scan: bool = False) -> Optional[pd.DataFrame]:
    # This function is correct, no changes needed
    logging.info(f"Fetching data for {symbol} ({timeframe})...")
    longest_period = max(params.get('long_window',0), params.get('price_ma_filter_period',0), params.get('volume_ma_period',0), params.get('rsi_period',0), params.get('atr_period',0), params.get('custom_slope_ma_period',0))
    try:
        if asset_type == 'stocks':
            yf_timeframe = timeframe.replace('d', 'd').replace('h', 'h').replace('m', 'm')
            period_to_fetch = "5y" if historical_scan else f"{longest_period * 3}d" if longest_period > 0 else "60d"
            df = yf.download(tickers=symbol, period=period_to_fetch, interval=yf_timeframe, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        elif asset_type == 'crypto':
            config = load_json('config.json')
            exchange_id = config.get('ccxt_exchange_id', 'binance')
            exchange = getattr(ccxt, exchange_id)()
            if not exchange.has['fetchOHLCV']: return None
            limit = 1000 if historical_scan else (longest_period + 50) if longest_period > 0 else 100
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv: return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else: return None
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')
        if len(df) < longest_period: logging.warning(f"Insufficient data for {symbol} for indicator warmup ({longest_period}): got {len(df)} rows.")
        return df.astype(float, errors='ignore')
    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {e}", exc_info=True)
        return None

def calculate_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    p = params
    df_copy = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_copy.columns: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(subset=['close'], inplace=True)
    if df_copy.empty: return df_copy

    def get_ma(source, length, ma_type):
        ma_type = ma_type.lower() if ma_type else 'sma'
        if length is None or length <= 0: return None
        # Ensure source has enough non-NaN values for the calculation
        if source.count() < length: return None
        if ma_type == 'sma': return ta.sma(source, length=length)
        if ma_type == 'ema': return ta.ema(source, length=length)
        return None

    # --- THIS IS THE CORRECTED LOGIC ---
    ma_short = get_ma(df_copy['close'], p.get('short_window'), p.get('ma_type'))
    if ma_short is not None:
        df_copy['ma_short'] = ma_short
        df_copy['ma_short_slope'] = ma_short.diff()

    ma_long = get_ma(df_copy['close'], p.get('long_window'), p.get('ma_type'))
    if ma_long is not None:
        df_copy['ma_long'] = ma_long
        df_copy['ma_long_slope'] = ma_long.diff()
        
    if p.get('use_price_ma_filter'):
        filter_ma = get_ma(df_copy['close'], p.get('price_ma_filter_period'), p.get('price_ma_filter_type'))
        if filter_ma is not None: df_copy['filter_ma'] = filter_ma

    if p.get('use_rsi_filter'):
        rsi_period = p.get('rsi_period')
        if rsi_period and df_copy['close'].count() >= rsi_period:
             df_copy['rsi'] = ta.rsi(df_copy['close'], length=rsi_period)

    if p.get('use_volatility_filter'):
        atr_period = p.get('atr_period')
        if atr_period and df_copy['high'].count() >= atr_period:
            df_copy['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=atr_period)

    if p.get('use_volume_filter') and 'volume' in df_copy.columns:
        volume_ma_period = p.get('volume_ma_period')
        if volume_ma_period and df_copy['volume'].count() >= volume_ma_period:
            df_copy['volume_ma'] = ta.sma(df_copy['volume'], length=volume_ma_period)

    if p.get('slope_filter_mode') == "Gunakan MA Kustom":
        custom_ma = get_ma(df_copy['close'], p.get('custom_slope_ma_period'), p.get('custom_slope_ma_type'))
        if custom_ma is not None:
            df_copy['custom_slope_ma'] = custom_ma
            df_copy['custom_slope_ma_slope'] = custom_ma.diff()
            
    return df_copy
# ---> THIS ENTIRE FUNCTION IS REWRITTEN AND CORRECTED <---
def evaluate_buy_filters(candle: pd.Series, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Evaluates all BUY-side filters with corrected logic.
    This version is simpler, more direct, and bug-free.
    """
    p = params
    failed_filters = []

    # Filter: Price vs Filter MA
    if p.get('use_price_ma_filter'):
        if candle.get('close', float('inf')) >= candle.get('filter_ma', 0):
            failed_filters.append('Price vs Filter MA')

    # Filter: RSI
    if p.get('use_rsi_filter'):
        # The condition is rsi < threshold. So it fails if rsi >= threshold.
        if candle.get('rsi', 100) >= p.get('rsi_buy_threshold', 50):
            failed_filters.append('RSI Buy')

    # Filter: Price Distance from MA
    if p.get('use_price_dist_filter'):
        # The condition is distance > threshold. So it fails if distance <= threshold.
        ma_short = candle.get('ma_short')
        if ma_short is not None and ma_short > 0:
            distance = (abs(candle.get('close', 0) - ma_short) / ma_short) * 100
            if distance <= p.get('price_dist_threshold', 0):
                failed_filters.append('Price Distance')
        else:
            # If ma_short is invalid, the filter automatically fails.
            failed_filters.append('Price Distance')

    # Filter: Spread between MAs
    if p.get('use_spread_filter'):
        # The condition is spread > threshold. So it fails if spread <= threshold.
        ma_long = candle.get('ma_long')
        ma_short = candle.get('ma_short')
        if ma_long is not None and ma_long > 0 and ma_short is not None:
            spread = (abs(ma_short - ma_long) / ma_long)
            if spread <= p.get('spread_threshold', 0):
                failed_filters.append('Spread')
        else:
            failed_filters.append('Spread')
            
    # Filter: Volatility (ATR)
    if p.get('use_volatility_filter'):
        # The condition is atr > min_value. So it fails if atr <= min_value.
        if candle.get('atr', 0) <= p.get('atr_min_value', 0):
            failed_filters.append('Volatility (ATR)')

    # Filter: Volume
    if p.get('use_volume_filter'):
        # The condition is volume > volume_ma. So it fails if volume <= volume_ma.
        if candle.get('volume', 0) <= candle.get('volume_ma', 0):
            failed_filters.append('Volume')

    # Filter: Slope
    slope_mode = p.get('slope_filter_mode', 'Nonaktif')
    if slope_mode != "Nonaktif":
        slope_to_check = None
        if slope_mode == "Gunakan MA Strategi":
            source = p.get('strategy_ma_source', 'MA Cepat')
            slope_to_check = candle.get('ma_short_slope') if source == "MA Cepat" else candle.get('ma_long_slope')
        elif slope_mode == "Gunakan MA Kustom":
            slope_to_check = candle.get('custom_slope_ma_slope')
        
        # The condition is slope >= threshold. So it fails if slope < threshold.
        if slope_to_check is None or slope_to_check < p.get('slope_threshold', 0):
            failed_filters.append('Slope Filter')
            
    all_passed = len(failed_filters) == 0
    return all_passed, failed_filters