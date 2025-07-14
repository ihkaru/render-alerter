# Final Google Colab Optimizer with Diagnostic Logging

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import optuna
import ccxt
from google.colab import files
import importlib
import sys
from collections import Counter

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('optimization_log.txt')
    ],
    force=True
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

def load_json(filepath: str):
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

# --- UPLOAD AND MODULE LOADING ---
print("Please upload your `optimizer_config.json`, `config.json`, `backtester.py`, and `shared_logic.py` files.")
uploaded = files.upload()
for filename, content in uploaded.items():
    with open(filename, 'wb') as f: f.write(content)

try:
    import backtester
    import shared_logic
    importlib.reload(shared_logic)
    importlib.reload(backtester)
    print("✅ Successfully imported 'backtester.py' and 'shared_logic.py'.")
except ImportError as e:
    logging.error(f"❌ Failed to import modules. Error: {e}")
    raise

# --- DATA HANDLING AND CACHING ---
GLOBAL_RAW_DATA_CACHE = {}

def fetch_raw_data_once(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    cache_key = (symbol, timeframe)
    if cache_key in GLOBAL_RAW_DATA_CACHE:
        return GLOBAL_RAW_DATA_CACHE[cache_key]

    logger.info(f"CACHE MISS: Performing deep historical fetch for {symbol} ({timeframe}). This will be slow ONCE per timeframe.")
    try:
        exchange = ccxt.binanceus({'options': {'defaultType': 'spot'}, 'hostname': 'api.binance.vision', 'enableRateLimit': True})
        since = int((datetime.now() - timedelta(days=365*5)).timestamp() * 1000)
        all_ohlcv = []
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        if not all_ohlcv: return None

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset='timestamp', inplace=True)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')

        GLOBAL_RAW_DATA_CACHE[cache_key] = df.astype(float, errors='ignore')
        return GLOBAL_RAW_DATA_CACHE[cache_key]
    except Exception as e:
        logging.error(f"Deep historical fetch for {symbol} failed: {e}", exc_info=True)
        return None

# --- OPTIMIZER LOGIC ---
FULL_CONFIG_GLOBAL = {}
OPTIMIZER_CONFIG_GLOBAL = {}

def calculate_robustness_score(trial: optuna.Trial, regime_results: List[dict], num_regimes: int) -> float:
    avg_pf, avg_wr, avg_dd, avg_trades_per_month, final_score = 0.0, 0.0, 100.0, 0.0, 0.0

    if len(regime_results) == num_regimes:
        is_viable = all(res.get('profit_factor', 0) > 1.0 and res.get('max_drawdown_pct', 100) < 99 for res in regime_results)
        if is_viable:
            avg_pf = np.mean([res['profit_factor'] for res in regime_results])
            avg_wr = np.mean([res['win_rate'] for res in regime_results])
            avg_dd = np.mean([res['max_drawdown_pct'] for res in regime_results])
            avg_trades_per_month = np.mean([res.get('trades_per_month', 0) for res in regime_results])
            performance_score = (avg_pf * (avg_wr / 100.0)) / (avg_dd + 1)
            mu, sigma = 19, 7
            frequency_score = np.exp(-0.5 * ((avg_trades_per_month - mu) / sigma) ** 2)
            final_score = performance_score * frequency_score

    trial.set_user_attr("win_rate", avg_wr)
    trial.set_user_attr("max_drawdown", avg_dd)
    trial.set_user_attr("trades_per_month", avg_trades_per_month)
    trial.set_user_attr("profit_factor", avg_pf)
    return final_score if np.isfinite(final_score) else 0.0

def objective(trial: optuna.Trial) -> float:
    params = FULL_CONFIG_GLOBAL['default_strategy_params'].copy()
    timeframe = trial.suggest_categorical("timeframe", OPTIMIZER_CONFIG_GLOBAL.get('timeframe_space'))
    for p_def in OPTIMIZER_CONFIG_GLOBAL.get('parameter_space', []):
        name = p_def['name']
        if 'condition' in p_def and not params.get(p_def['condition']): continue
        p_type = p_def['type']
        if p_type == 'categorical': params[name] = trial.suggest_categorical(name, p_def['choices'])
        elif p_type == 'int': params[name] = trial.suggest_int(name, p_def['low'], p_def['high'], step=p_def.get('step', 1))
        else: params[name] = trial.suggest_float(name, p_def['low'], p_def['high'], step=p_def.get('step', 0.1))

    if params.get('short_window', 1) >= params.get('long_window', 2):
        trial.set_user_attr("win_rate", 0); trial.set_user_attr("max_drawdown", 100)
        trial.set_user_attr("trades_per_month", 0); trial.set_user_attr("profit_factor", 0)
        return -2.0

    raw_df = fetch_raw_data_once(FULL_CONFIG_GLOBAL['symbol'], timeframe)
    if raw_df is None or raw_df.empty: return -3.0

    df_with_indicators = shared_logic.calculate_indicators(raw_df, params)
    
    regime_results = []
    all_backtest_results = [] # Store all results for debugging

    for regime_name, dates in OPTIMIZER_CONFIG_GLOBAL.get('market_regimes', {}).items():
        regime_df = df_with_indicators.loc[dates['start_date']:dates['end_date']].copy()
        if regime_df.empty: continue

        config = {'strategy_params': params, 'backtest_settings': FULL_CONFIG_GLOBAL['backtest_settings']}
        
        result = backtester.run_backtest(
            symbol=FULL_CONFIG_GLOBAL['symbol'], asset_type=FULL_CONFIG_GLOBAL['asset_type'],
            timeframe=timeframe, full_config=config, preloaded_data=regime_df
        )
        all_backtest_results.append({'regime': regime_name, 'result': result})

        if "metrics" in result and result["metrics"] and result["metrics"].get("total_trades", 0) > 2:
            duration_days = (pd.to_datetime(dates['end_date']) - pd.to_datetime(dates['start_date'])).days
            trades_per_month = (result["metrics"]["total_trades"] / duration_days) * 30 if duration_days > 0 else 0
            result["metrics"]["trades_per_month"] = trades_per_month
            regime_results.append(result["metrics"])
    
    # --- NEW: DIAGNOSTIC LOGGING ---
    # If after checking all regimes, none were profitable, log why.
    if not regime_results and all_backtest_results:
        # Check the debug info from the first regime's backtest
        first_result = all_backtest_results[0]['result']
        first_regime_name = all_backtest_results[0]['regime']
        debug_info = first_result.get('debug_info', {})
        
        if debug_info:
            crossovers = debug_info.get('crossover_events_found', 0)
            failed_counts = debug_info.get('failed_filter_counts', {})
            
            top_fail_reason = "N/A"
            if failed_counts:
                top_fail_reason = Counter(failed_counts).most_common(1)[0][0]

            logger.warning(
                f"Trial {trial.number:03} failed all regimes. "
                f"Diagnostic (from '{first_regime_name}'): "
                f"Crossovers found: {crossovers}. Top failing filter: '{top_fail_reason}'"
            )
    # --- END OF NEW LOGGING ---

    return calculate_robustness_score(trial, regime_results, len(OPTIMIZER_CONFIG_GLOBAL.get('market_regimes', {})))

def print_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    win_rate = trial.user_attrs.get("win_rate", 0.0)
    max_dd = trial.user_attrs.get("max_drawdown", 100.0)
    trades_month = trial.user_attrs.get("trades_per_month", 0.0)
    profit_factor = trial.user_attrs.get("profit_factor", 0.0)
    score = trial.value if trial.value is not None else 0.0
    best_score = study.best_value if study.best_value is not None else 0.0

    log_message = (
        f"Trial {trial.number:03} | " f"Score: {score:7.4f} | " f"Best Score: {best_score:7.4f} | "
        f"Win Rate: {win_rate:5.1f}% | " f"Max DD: {max_dd:5.1f}% | " f"Trades/Mo: {trades_month:5.1f} | "
        f"Profit Factor: {profit_factor:5.2f}"
    )
    logger.info(log_message)

def run_optimization_flow(symbol: str, asset_type: str) -> pd.DataFrame:
    global FULL_CONFIG_GLOBAL, OPTIMIZER_CONFIG_GLOBAL
    config = load_json('config.json')
    optimizer_config = load_json('optimizer_config.json')
    if not all([optimizer_config.get('market_regimes'), optimizer_config.get('timeframe_space')]):
        logging.error("❌ `optimizer_config.json` must have 'market_regimes' and 'timeframe_space'.")
        return pd.DataFrame()
    FULL_CONFIG_GLOBAL = {'symbol': symbol, 'asset_type': asset_type, **config}
    OPTIMIZER_CONFIG_GLOBAL = optimizer_config

    logger.info("="*80)
    logger.info(f"Starting new optimization study for {symbol}")
    logger.info(f"Number of trials: {optimizer_config.get('optimizer_settings', {}).get('n_trials', 100)}")
    logger.info("="*80)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=optimizer_config.get('optimizer_settings', {}).get('n_trials', 100),
        callbacks=[print_progress_callback]
    )

    if not study.trials: return pd.DataFrame()
    df = study.trials_dataframe()
    df['win_rate'] = df['user_attrs'].apply(lambda x: x.get('win_rate', 0))
    df['max_drawdown'] = df['user_attrs'].apply(lambda x: x.get('max_drawdown', 100))
    df['trades_per_month'] = df['user_attrs'].apply(lambda x: x.get('trades_per_month', 0))
    df['profit_factor'] = df['user_attrs'].apply(lambda x: x.get('profit_factor', 0))
    df = df[(df.state == "COMPLETE") & (df.value > 0)].sort_values(by="value", ascending=False)
    if df.empty: return pd.DataFrame()
    
    param_cols = [c for c in df.columns if c.startswith('params_')]
    metric_cols = ['value', 'win_rate', 'max_drawdown', 'trades_per_month', 'profit_factor']
    results = df[metric_cols + param_cols].head(5)
    results.rename(columns={'value': 'Robustness_Score'}, inplace=True)
    results.columns = [c.replace('params_', '') for c in results.columns]
    results = results.round({'Robustness_Score': 4, 'win_rate': 2, 'max_drawdown': 2, 'trades_per_month': 1, 'profit_factor': 3})
    return results.reset_index(drop=True)

# --- EXECUTION FLOW ---
symbol = "BTC/USDT" #@param {type:"string"}
asset_type = "crypto" #@param ["crypto", "stocks"]

print("\n\nStarting fully-validated robust optimization...")
best_params_df = run_optimization_flow(symbol, asset_type)

if not best_params_df.empty:
    results_json = best_params_df.to_json(orient='records', indent=2)
    print("\n\n" + "="*50)
    print("✅ Robust Optimization Complete!")
    print("📋 Top 5 Parameter Sets:")
    display(best_params_df)
    with open('best_params.json', 'w') as f: f.write(results_json)
    print("\n\n✅ Results saved to 'best_params.json'.")
else:
    print("\n\n" + "="*50)
    print("❌ Optimization did not produce any robust results.")