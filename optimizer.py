import optuna
import pandas as pd
from typing import Dict, Any, List
from backtester import run_backtest
from shared_logic import load_json

FULL_CONFIG_GLOBAL = {}

def calculate_score(metrics: Dict[str, Any]) -> float:
    if not metrics or metrics.get('total_trades', 0) < 3:
        return -1.0 

    profit_factor = metrics.get('profit_factor', 0)
    win_rate = metrics.get('win_rate', 0) / 100.0 
    max_drawdown_pct = metrics.get('max_drawdown_pct', 100)
    
    if max_drawdown_pct == 0: max_drawdown_pct = 0.01

    score = (profit_factor * win_rate) / max_drawdown_pct
    return score if pd.notna(score) else -1.0

def objective(trial: optuna.Trial) -> float:
    params = FULL_CONFIG_GLOBAL['strategy_params'].copy()
    optimizer_config = load_json('optimizer_config.json', {})
    
    timeframe = trial.suggest_categorical("timeframe", optimizer_config.get('timeframe_space', ["1h"]))
    date_range_space = optimizer_config.get('date_range_space', {})
    date_range_name = trial.suggest_categorical("date_range_name", list(date_range_space.keys()))
    start_date = date_range_space[date_range_name]['start_date']
    end_date = date_range_space[date_range_name]['end_date']
    
    # --- NEW: Conditional Parameter Suggestion ---
    param_space = optimizer_config.get('parameter_space', [])
    for p_def in param_space:
        name = p_def['name']
        # If the parameter has a condition, check if the condition is met
        if 'condition' in p_def:
            condition_param = p_def['condition']
            # We need to suggest the condition param first if it hasn't been suggested
            if condition_param not in params:
                 # Find the definition for the condition parameter
                 cond_def = next((p for p in param_space if p['name'] == condition_param), None)
                 if cond_def:
                     params[condition_param] = trial.suggest_categorical(condition_param, cond_def['choices'])

            # Only suggest this parameter if its controlling condition is True
            if not params.get(condition_param):
                continue

        # Suggest the value based on type
        if p_def['type'] == 'int':
            params[name] = trial.suggest_int(name, p_def['low'], p_def['high'], step=p_def.get('step', 1))
        elif p_def['type'] == 'float':
            params[name] = trial.suggest_float(name, p_def['low'], p_def['high'], step=p_def.get('step', 0.1))
        elif p_def['type'] == 'categorical':
             params[name] = trial.suggest_categorical(name, p_def['choices'])

    if params.get('short_window', 1) >= params.get('long_window', 2):
        return -1.0

    live_config = FULL_CONFIG_GLOBAL.copy()
    live_config['strategy_params'] = params
    live_config['strategy_start_timestamp'] = start_date
    live_config['strategy_end_timestamp'] = end_date

    results = run_backtest(
        symbol=live_config['symbol'],
        asset_type=live_config['asset_type'],
        timeframe=timeframe,
        full_config=live_config
    )
    
    if "error" in results or "message" in results:
        return -1.0
        
    return calculate_score(results['metrics'])

def run_optimization(symbol: str, asset_type: str, full_config: Dict[str, Any]) -> pd.DataFrame:
    global FULL_CONFIG_GLOBAL
    FULL_CONFIG_GLOBAL = full_config.copy()
    FULL_CONFIG_GLOBAL['symbol'] = symbol
    FULL_CONFIG_GLOBAL['asset_type'] = asset_type
    
    optimizer_config = load_json('optimizer_config.json', {})
    n_trials = optimizer_config.get('optimizer_settings', {}).get('n_trials', 100)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    trials_df = study.trials_dataframe()
    # Ensure value column is numeric and handle potential errors
    trials_df['value'] = pd.to_numeric(trials_df['value'], errors='coerce')
    completed_df = trials_df[trials_df.state == "COMPLETE"].dropna(subset=['value']).sort_values(by="value", ascending=False)
    
    param_columns = [col for col in completed_df.columns if col.startswith('params_')]
    results_df = completed_df[['value'] + param_columns].head(5)
    
    results_df.rename(columns={'value': 'Score'}, inplace=True)
    results_df.columns = [col.replace('params_', '') for col in results_df.columns]
    
    return results_df.reset_index(drop=True)