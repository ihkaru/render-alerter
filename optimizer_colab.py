# Final, Definitive Google Colab Optimizer with Per-Trial and Per-Regime Parameter Logging

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('optimization_log.txt')], force=True)
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
    import backtester; import shared_logic
    importlib.reload(shared_logic); importlib.reload(backtester)
    print("✅ Successfully imported 'backtester.py' and 'shared_logic.py'.")
except ImportError as e:
    logging.error(f"❌ Failed to import modules. Error: {e}"); raise

# --- DATA FETCHER ---
GLOBAL_RAW_DATA_CACHE = {}
def fetch_raw_data_once(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    cache_key = (symbol, timeframe)
    if cache_key in GLOBAL_RAW_DATA_CACHE: return GLOBAL_RAW_DATA_CACHE[cache_key]
    logger.info(f"CACHE MISS: Performing deep historical fetch for {symbol} ({timeframe}).")
    try:
        exchange = ccxt.binanceus({'options': {'defaultType': 'spot'}, 'hostname': 'api.binance.vision', 'enableRateLimit': True})
        since = int((datetime.now() - timedelta(days=365*5)).timestamp() * 1000)
        all_ohlcv = []; last_timestamp = -1
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            if last_timestamp == ohlcv[-1][0]: logger.error(f"Infinite loop detected! Aborting."); break
            last_timestamp = ohlcv[-1][0]; all_ohlcv.extend(ohlcv); since = ohlcv[-1][0] + 1
        if not all_ohlcv: logger.error(f"No data fetched for {symbol}."); return None
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset='timestamp', inplace=True); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        GLOBAL_RAW_DATA_CACHE[cache_key] = df.astype(float, errors='ignore'); return GLOBAL_RAW_DATA_CACHE[cache_key]
    except Exception as e:
        logging.error(f"Data fetch failed: {e}", exc_info=True); return None

# --- OPTIMIZER LOGIC ---
FULL_CONFIG_GLOBAL, OPTIMIZER_CONFIG_GLOBAL = {}, {}

def calculate_robustness_score(trial: optuna.Trial, successful_regimes: List[dict], total_regimes: int, regime_details: dict) -> float:
    if not successful_regimes:
        final_score = 0.0
    else:
        avg_pf = np.mean([res['profit_factor'] for res in successful_regimes])
        avg_wr = np.mean([res['win_rate'] for res in successful_regimes])
        avg_dd = np.mean([res['max_drawdown_pct'] for res in successful_regimes])
        avg_trades_per_month = np.mean([res.get('trades_per_month', 0) for res in successful_regimes])
        avg_sharpe = np.mean([res.get('sharpe_ratio', 0) for res in successful_regimes])
        avg_sortino = np.mean([res.get('sortino_ratio', 0) for res in successful_regimes])

        capped_pf = min(avg_pf, 5.0)
        safe_dd = max(avg_dd, 0.5)
        risk_adjusted_return = max(avg_sortino, avg_sharpe) if avg_sortino > 0 else avg_sharpe
        risk_adjusted_return = max(risk_adjusted_return, 0.1)
        base_performance = (capped_pf * (avg_wr / 100.0)) / (safe_dd / 100.0 + 1)
        risk_factor = 1 + (risk_adjusted_return * 0.5)
        mu, sigma = 13, 7
        frequency_score = np.exp(-0.5 * ((avg_trades_per_month - mu) / sigma) ** 2)
        robustness_penalty = len(successful_regimes) / total_regimes
        final_score = base_performance * risk_factor * frequency_score * robustness_penalty

        if avg_sharpe == 0.0 and avg_sortino == 0.0: final_score *= 0.3
        if avg_trades_per_month < 1.0: final_score *= 0.5
        if avg_dd < 0.1: final_score *= 0.4

    trial.set_user_attr("win_rate", float(np.mean([r.get('win_rate', 0) for r in successful_regimes])) if successful_regimes else 0.0)
    trial.set_user_attr("max_drawdown", float(np.mean([r.get('max_drawdown_pct', 100) for r in successful_regimes])) if successful_regimes else 100.0)
    trial.set_user_attr("trades_per_month", float(np.mean([r.get('trades_per_month', 0) for r in successful_regimes])) if successful_regimes else 0.0)
    trial.set_user_attr("profit_factor", float(np.mean([r.get('profit_factor', 0) for r in successful_regimes])) if successful_regimes else 0.0)
    trial.set_user_attr("sharpe_ratio", float(np.mean([r.get('sharpe_ratio', 0) for r in successful_regimes])) if successful_regimes else 0.0)
    trial.set_user_attr("sortino_ratio", float(np.mean([r.get('sortino_ratio', 0) for r in successful_regimes])) if successful_regimes else 0.0)

    try:
        sanitized_regime_details = json.loads(json.dumps(regime_details, default=float))
        trial.set_user_attr("regime_breakdown", sanitized_regime_details)
    except Exception as e:
        logger.error(f"Failed to serialize regime_breakdown for Trial {trial.number}: {e}")
        trial.set_user_attr("regime_breakdown", {"error": "serialization failed"})

    return final_score if np.isfinite(final_score) else 0.0

def objective(trial: optuna.Trial) -> float:
    params = FULL_CONFIG_GLOBAL['default_strategy_params'].copy()
    short_window_def = next((p for p in OPTIMIZER_CONFIG_GLOBAL['parameter_space'] if p['name'] == 'short_window'), None)
    long_window_def = next((p for p in OPTIMIZER_CONFIG_GLOBAL['parameter_space'] if p['name'] == 'long_window'), None)
    params['short_window'] = trial.suggest_int('short_window', short_window_def['low'], short_window_def['high'], step=short_window_def.get('step', 1))
    long_window_low = max(long_window_def['low'], params['short_window'] + 1)
    if long_window_low >= long_window_def['high']: return -2.0
    params['long_window'] = trial.suggest_int('long_window', long_window_low, long_window_def['high'], step=long_window_def.get('step', 1))
    for p_def in OPTIMIZER_CONFIG_GLOBAL.get('parameter_space', []):
        name = p_def['name']
        if name in ['short_window', 'long_window']: continue
        if 'condition' in p_def and not trial.params.get(p_def['condition'], False) and not params.get(p_def['condition'], False): continue
        p_type = p_def['type']
        if p_type == 'categorical': params[name] = trial.suggest_categorical(name, p_def['choices'])
        elif p_type == 'int': params[name] = trial.suggest_int(name, p_def['low'], p_def['high'], step=p_def.get('step', 1))
        else: params[name] = trial.suggest_float(name, p_def['low'], p_def['high'], step=p_def.get('step', 0.1))
    timeframe = trial.suggest_categorical("timeframe", OPTIMIZER_CONFIG_GLOBAL.get('timeframe_space'))
    raw_df = fetch_raw_data_once(FULL_CONFIG_GLOBAL['symbol'], timeframe)
    if raw_df is None or raw_df.empty: return -3.0
    df_with_indicators = shared_logic.calculate_indicators(raw_df, params)

    successful_regimes, regime_configs = [], OPTIMIZER_CONFIG_GLOBAL.get('market_regimes', {})
    regime_details_for_logging = {}

    for regime_name, dates in regime_configs.items():
        regime_df = df_with_indicators.loc[dates['start_date']:dates['end_date']].copy()
        if regime_df.empty: continue

        result = backtester.run_backtest(
            symbol=FULL_CONFIG_GLOBAL['symbol'],
            asset_type=FULL_CONFIG_GLOBAL['asset_type'],
            timeframe=timeframe,
            full_config={'strategy_params': params, **FULL_CONFIG_GLOBAL},
            preloaded_data=regime_df
        )

        if "error" in result: continue
        metrics = result.get("metrics", {})
        metrics = validate_and_sanitize_metrics(metrics)

        if metrics:
            duration_days = (pd.to_datetime(dates['end_date']) - pd.to_datetime(dates['start_date'])).days
            metrics["trades_per_month"] = (metrics.get("total_trades", 0) / duration_days) * 30 if duration_days > 0 else 0

            # --- MODIFICATION: Calculate and add Total PnL ---
            total_pnl = 0
            if "trades" in result and result["trades"]:
                total_pnl = sum(trade['profit'] for trade in result['trades'])
            metrics["total_pnl"] = total_pnl
            # --- END MODIFICATION ---

            regime_details_for_logging[regime_name] = metrics
            if metrics.get("total_trades", 0) > 2 and metrics.get("profit_factor", 0) > 1.0:
                successful_regimes.append(metrics)

    return calculate_robustness_score(trial, successful_regimes, len(regime_configs), regime_details_for_logging)

def validate_and_sanitize_metrics(metrics: dict) -> dict:
    sanitized = metrics.copy()
    sortino = metrics.get('sortino_ratio', 0)
    if sortino > 5.0: sanitized['sortino_ratio'] = 5.0
    elif sortino < -5.0: sanitized['sortino_ratio'] = -5.0
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 4.0: sanitized['sharpe_ratio'] = 4.0
    elif sharpe < -4.0: sanitized['sharpe_ratio'] = -4.0
    pf = metrics.get('profit_factor', 0)
    if pf > 10.0: sanitized['profit_factor'] = 10.0
    return sanitized

def print_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    win_rate = trial.user_attrs.get("win_rate", 0.0)
    max_dd = trial.user_attrs.get("max_drawdown", 100.0)
    trades_month = trial.user_attrs.get("trades_per_month", 0.0)
    profit_factor = trial.user_attrs.get("profit_factor", 0.0)
    sharpe = trial.user_attrs.get("sharpe_ratio", 0.0)
    sortino = trial.user_attrs.get("sortino_ratio", 0.0)
    score = trial.value if trial.value is not None else 0.0
    best_score = study.best_value if study.best_value is not None else -1.0

    log_message = (f"TRIAL {trial.number:03} | SCORE: {score:7.4f} | Best: {best_score:7.4f} | "
                   f"Avg Win Rate: {win_rate:5.1f}% | Avg Max DD: {max_dd:5.1f}% | "
                   f"Avg Trades/Mo: {trades_month:5.1f} | Avg PF: {profit_factor:5.2f} | "
                   f"Avg Sharpe: {sharpe:5.2f} | Avg Sortino: {sortino:5.2f}")
    logger.info(log_message)

    regime_details = trial.user_attrs.get("regime_breakdown", {})
    if regime_details:
        logger.info("    > Regime Breakdown:")
        for regime, m in regime_details.items():
            # --- MODIFICATION: Added PnL to the log string ---
            regime_log = (f"      - {regime:<22}: PnL=${m.get('total_pnl', 0):>8.2f}, WR={m.get('win_rate', 0):5.1f}%, "
                          f"MDD={m.get('max_drawdown_pct', 0):5.1f}%, TPM={m.get('trades_per_month', 0):4.1f}, "
                          f"Sharpe={m.get('sharpe_ratio', 0):5.2f}, Sortino={m.get('sortino_ratio', 0):5.2f}")
            logger.info(regime_log)
            # --- END MODIFICATION ---

    params_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])
    logger.info(f"    > PARAMS: {params_str}")

def run_optimization_flow(symbol: str, asset_type: str) -> pd.DataFrame:
    global FULL_CONFIG_GLOBAL, OPTIMIZER_CONFIG_GLOBAL
    config = load_json('config.json'); optimizer_config = load_json('optimizer_config.json')
    if not all([optimizer_config.get('market_regimes'), optimizer_config.get('timeframe_space')]):
        logging.error("Missing config keys."); return pd.DataFrame()
    FULL_CONFIG_GLOBAL = {'symbol': symbol, 'asset_type': asset_type, **config}; OPTIMIZER_CONFIG_GLOBAL = optimizer_config
    logger.info("="*80); logger.info(f"Starting Optimization for {symbol}"); logger.info("="*80)
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(
            objective,
            n_trials=optimizer_config.get('optimizer_settings', {}).get('n_trials', 5000),
            n_jobs=-1,
            callbacks=[print_progress_callback]
        )
    except Exception as e:
        logger.error(f"An unhandled exception occurred during study.optimize: {e}", exc_info=True)
        return pd.DataFrame()

    # --- START: MODIFIED AND MORE ROBUST RESULTS COLLECTION ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0]

    if not completed_trials:
        logger.warning("No completed trials produced a positive score. Cannot generate final results.")
        return pd.DataFrame()

    results_data = []
    for trial in completed_trials:
        # Combine parameters and the user attributes into one dictionary
        row = trial.params.copy()
        row['Robustness_Score'] = trial.value
        row.update(trial.user_attrs)
        results_data.append(row)

    df = pd.DataFrame(results_data)
    df = df.sort_values(by="Robustness_Score", ascending=False)
    # --- END: MODIFIED AND MORE ROBUST RESULTS COLLECTION ---

    if df.empty: return pd.DataFrame()

    # Keep only the relevant columns for the final output
    metric_cols = ['Robustness_Score', 'win_rate', 'max_drawdown', 'trades_per_month', 'profit_factor', 'sharpe_ratio', 'sortino_ratio']
    param_cols = list(study.best_trial.params.keys()) # Get param names from a sample trial
    all_display_cols = metric_cols + param_cols
    # Filter df to only include columns that actually exist
    final_cols = [col for col in all_display_cols if col in df.columns]
    results = df[final_cols].head(5)

    results = results.round({
        'Robustness_Score': 4, 'win_rate': 2, 'max_drawdown': 2, 'trades_per_month': 1,
        'profit_factor': 3, 'sharpe_ratio': 3, 'sortino_ratio': 3
    })
    return results.reset_index(drop=True)


# --- EXECUTION FLOW ---
symbol = "ETH/USDT"; asset_type = "crypto"
print("\n\nStarting fully-validated robust optimization...")
best_params_df = run_optimization_flow(symbol, asset_type)
if not best_params_df.empty:
    results_json = best_params_df.to_json(orient='records', indent=2)
    print("\n\n" + "="*50);
    print("✅ Robust Optimization Complete!");
    print("📋 Top 5 Parameter Sets (DataFrame View):")
    from IPython.display import display; display(best_params_df)
    with open('best_params.json', 'w') as f: f.write(results_json)
    print("\n✅ Results saved to 'best_params.json'.")

    print("\n\n" + "="*70)
    print("⬇️⬇️⬇️  COPY THE JSON BLOCK BELOW AND PASTE IT INTO THE STREAMLIT APP ⬇️⬇️⬇️")
    print("="*70)
    print(results_json)
    print("="*70)

else:
    print("\n\n" + "="*50);
    print("❌ Optimization did not produce any robust results.")