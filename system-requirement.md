# System Requirements Document: Personal Trading Alerter & Backtester

**Version:** 6.0  
**Date:** July 20, 2025  
**Author:** Gemini & User

**Revision Note:** This major version introduces a high-performance backtesting engine accelerated with Numba JIT compilation. The optimization workflow has been completely redesigned to use a sophisticated, cloud-based (Google Colab) robustness tester that evaluates parameter sets across multiple, pre-defined market regimes. This version also adds more detailed performance metrics (Sharpe/Sortino Ratios), implements data caching, and clarifies the logic for all buy-side and sell-side filters.

## 1. Project Vision & Purpose

This document outlines the system requirements for a personal, professional-grade quantitative trading analysis and alerting tool.

The primary purpose of this system is to provide its user with a reliable, non-repainting, and high-performance backtesting engine, alongside a persistent live alerting mechanism. It is designed to be a fully-owned alternative to cloud platforms, giving the user complete control over the strategy logic, data sources, and operational environment. The overarching goal is to enable rapid strategy iteration and validation to support the user's investment portfolio growth. Furthermore, it introduces an automated, cloud-based robustness optimization engine to systematically discover durable strategy parameters that perform well across varied market conditions.

## 2. Core Philosophy & Guiding Principles

The system is built upon several core principles that must be respected in all future development:

### Non-Repainting Logic

All backtesting and alerting logic MUST avoid look-ahead bias. Decisions are made on the data of a fully closed "signal candle," and actions (trades, alerts) are executed on the open of the subsequent "action candle."

### Differentiated & Prioritized Exit Logic

The system's logic for entering and exiting positions is fundamentally different and MUST be respected.

#### Entry Logic (AND)

A trade is only entered if the primary signal (e.g., a Golden Cross) occurs AND it passes a sequential chain of all enabled buy-side filters.

#### Exit Logic (Prioritized OR)

A trade is exited if ANY of the following exit conditions are met, checked in a specific order of priority:

1. Stop Loss
2. Recovery Exit
3. Primary Sell Signal (which itself has its own mandatory filter conditions)

#### Primary Sell Signal (AND)

The primary sell signal (e.g., negative MA slope) is only valid if it occurs AND it passes its required sell-side filters (e.g., RSI is above its sell threshold).

### End-of-Period Position Closure

At the conclusion of a backtest, any open position MUST be automatically closed at the close price of the final available candle.

### State Persistence

The system must be resilient to restarts. The state of the live alerter and the list of user-defined alerts are persisted to disk.

### Configuration-Driven

The system's behavior is primarily defined by user-editable JSON configuration files.

### Testability and Validation

The system MUST be accompanied by a comprehensive suite of automated unit tests.

## 3. System Architecture & Technology Stack

- **Language:** Python 3.10+
- **Core Libraries:** pandas, numpy, numba, pandas-ta, ccxt, yfinance, streamlit, plotly, schedule, requests, optuna
- **Testing Stack:** pytest, pytest-mock
- **Containerization:** Docker & Docker Compose

## 4. Data Flow & State Management

- **Configuration:** config.json, alerts.json, optimizer_config.json
- **State:** state.json
- **Backtesting Workflow:** (Unchanged)
- **Alerting Workflow:** (Unchanged)
- **Optimization Workflow:** (Unchanged)

## 5. Functional Requirements (Features)

### F1: Strategy Backtesting Engine

#### F1.1

Must perform a stateful, non-repainting backtest as per the logic defined in section 2. The core loop MUST be JIT-compiled with Numba for high performance.

#### F1.2

Must support configurable commission costs, applied to both entry and exit of a trade.

#### F1.3 (Updated & Expanded) Comprehensive Filtering, Risk & Exit Management

The engine must correctly implement all of the following rules based on the user's configuration.

##### Buy-Side Filtering (AND Logic)

A primary buy signal is only actioned if it passes ALL enabled filters below:

- **Price vs Filter MA:** close price must be >= the filter_ma
- **RSI Buy:** rsi value must be < the rsi_buy_threshold
- **Price Distance:** The percentage distance between the close price and the short MA must be > the threshold
- **MA Spread:** The percentage spread between the short and long MAs must be > the threshold
- **Volatility (ATR):** The atr value must be > the minimum required value
- **Volume:** The candle's volume must be > its moving average (volume_ma)
- **Slope Filter:** The slope of the selected MA must be > the required threshold

##### Independent Exit Triggers (Prioritized OR Logic)

An exit is triggered by the first of the following conditions to be met:

1. **Stop Loss:** Must exit a trade at the exact stop-loss price if the low of any action candle touches or crosses below it
2. **Recovery Exit:** Must correctly track a trade that has dipped significantly and exit it if the price recovers to near the entry point, as defined by the thresholds
3. **Filtered Primary Sell Signal (AND Logic):** The primary sell signal (negative ma_short_slope) is only actioned if it passes all enabled sell-side filters
   - **RSI Sell Filter:** If enabled, the primary sell signal is only valid if the RSI on the signal candle is also >= the rsi_sell_threshold

##### Trade Management Rules

- **Minimum Hold Duration:** Must ignore a valid primary sell signal if the trade has not been open for the required number of bars
- **Size Down:** Must correctly reduce the position size for the next trade following a losing trade, based on the configured percentage
- **Fake Loss:** Must correctly calculate a separate adjusted_equity by "withdrawing" a percentage of profits, and use this reduced equity value for all subsequent position sizing calculations
- **Leverage Multiplier:** Must correctly scale the calculated entry_size_usd by the specified multiplier
- **Cooldown Period:** Must prevent a new buy signal from executing if it occurs within the configured number of bars after the previous trade's exit

#### F1.4 Timeframe-Aware Indicator Warm-up

Must fetch sufficient historical data and correctly align the backtest start to the first candle where all required indicators (based on the longest lookback period) are valid. The user's requested start date is a minimum; the actual start may be later.

#### F1.5 (Updated) Comprehensive Results Dictionary

Must output a results dictionary containing performance metrics, a detailed trade log, and equity curve data. It MUST always return the historical_data and debug_info objects, even if no trades were executed.

#### F1.6 Backtest Debugging Output

If a backtest results in zero trades, the debug_info object MUST contain the count of primary buy signals and a breakdown of which filters caused the rejections.

#### F1.7 End-of-Period Position Closure

If a position is still open when historical data is exhausted, the backtester MUST automatically close the trade at the close price of the final candle.

#### F1.8 Backtest over Specific Date Ranges

The backtesting engine must accept an optional end_date to allow for analysis of specific historical periods.

### F2: Live Alerting Service

#### F2.1

Must run as a continuous, scheduled background process.

#### F2.2

Must read its list of jobs from alerts.json.

#### F2.3 (Updated) Consistent Signal Logic

Must process each alert independently, implementing the same signal and filter logic as the backtester (including all buy-side filters and the prioritized Stop Loss / Recovery / Filtered Sell Signal exit logic) for maximum consistency.

#### F2.4

Must maintain a persistent state for each alert using state.json.

#### F2.5

Must integrate with the Telegram API.

### F3: Alert Management (CRUD)

#### F3.1 (Create)

The user must be able to create a new live alert from the UI. This action saves the currently configured symbol, timeframe, and strategy parameters to alerts.json.

#### F3.2 (Read)

The UI must display a list of all currently active alerts from alerts.json.

#### F3.3 (Delete)

The user must be able to remove an active alert via a "Remove" button in the UI.

### F4: Data & Configuration

#### F4.1

The UI must provide widgets for every single parameter present in the reference config.json.

#### F4.2

The UI must dynamically fetch and display a list of available crypto assets from the configured exchange.

### F5: Performance Analytics

#### F5.1 Buy and Hold Comparison

The backtester MUST calculate the full equity curve for a simple "Buy and Hold" strategy over the same period for comparison.

#### F5.2 (Updated) Advanced Risk & Return Metrics

The backtester MUST calculate the portfolio's drawdown series, identify the maximum drawdown value ($) and percentage (%), and also calculate the Sharpe Ratio and Sortino Ratio.

### F6: Strategy Optimization Engine (Cloud-Based Robustness Testing)

#### F6.1 Automated Parameter Search via Colab

The system MUST provide a Google Colab notebook (optimizer_colab.py) that uses the Optuna library to intelligently search for optimal strategy parameters.

#### F6.2 Configurable Search Space

The search space for the optimizer MUST be defined in optimizer_config.json.

#### F6.3 Mandatory Robustness Testing (Market Regimes)

The core function of the optimizer is to run each trial across multiple, pre-defined historical date ranges (e.g., "Bull_Run_2021", "Bear_Market_2022") to find parameters that are robust across various market conditions.

#### F6.4 Multi-Objective Robustness Scoring

The optimizer MUST evaluate each trial using a single, sophisticated score that positively weights profitability (Profit Factor), risk-adjusted returns (Sharpe/Sortino), and win rate, while also rewarding for a reasonable trade frequency and penalizing for inconsistency across the tested market regimes.

#### F6.5 Top Results Display

The optimizer notebook MUST output the top 5 best-performing parameter sets as a JSON object, including the final robustness score and key average performance metrics (win rate, drawdown, etc.), ready to be pasted into the UI.

#### F6.6 Optimizer UI Workflow

The main application UI must provide a dedicated area for the user to paste the JSON output from the Colab notebook to display the results and load them into the application's controls.

## 6. User Interface (UI) Requirements

The application UI is a web-based dashboard built with Streamlit.

### UR1: Layout

#### UR1.1 Sidebar Control Panel

All strategy configuration widgets and primary action buttons ("Run Backtest", "Add as Live Alert") MUST be located in a collapsible st.sidebar. An expander in the sidebar MUST provide instructions and a link for running the Cloud-Based Optimizer.

#### UR1.2 Main Display Area

The main area is dedicated to displaying either the detailed backtest results or the strategy optimization results table.

#### UR1.3 Collapsible Alerts Panel

The list of currently active alerts from alerts.json MUST be displayed in a collapsible st.expander at the bottom of the main page.

### UR2: Backtest Results

#### UR2.1 (KPIs)

Key Performance Indicators MUST be displayed prominently, including Net Profit, Win Rate, Max Drawdown, Profit Factor, Sharpe Ratio, and Sortino Ratio.

#### UR2.2 (Interactive Price & RSI Chart)

A stacked, two-panel Plotly chart must be displayed. Both panels MUST share a synchronized x-axis.

- The top panel MUST contain the price candlestick chart, trade markers, and moving averages
- The bottom panel MUST display the RSI indicator with its buy/sell threshold lines

#### UR2.3 (Equity Analysis Chart)

A dedicated, interactive Plotly chart must display the portfolio equity curve and drawdown.

#### UR2.4 (Debugging & No-Trade Display)

If a backtest produces no trades, the UI MUST still render the price chart and display the debugging information.

#### UR2.5 (Trade Log)

A collapsible table of every trade must be available. This table MUST include entry_rsi and exit_rsi columns.

### UR3: Optimization Results

#### UR3.1 Top 5 Results Table

The UI must display the results from the optimizer in a clear table, including the Robustness_Score and other key metrics like average win rate and drawdown.

#### UR3.2 Interactive Parameter Loading

Each row in the optimization results table MUST have a "Load" button/checkbox. Activating it MUST instantly update all corresponding control widgets in the sidebar to match the values from that row.

#### UR3.3 Colab JSON Input

The UI MUST feature a text area where a user can paste the JSON output generated by the optimizer_colab.py notebook. A button next to it will parse this JSON and display the results in the table (UR3.1).

## 7. Non-Functional Requirements

### NFR1 (Reliability)

The alerter.py service must be configured to restart automatically on failure.

### NFR2 (Security)

All sensitive credentials MUST be managed via environment variables.

### NFR3 (Usability)

The application must be fully runnable within a local Docker environment with clear instructions.

### NFR4 (Testability & Validation)

The project MUST maintain a suite of unit tests using pytest to validate core logic.

### NFR5: Robustness to Imperfect Data

The backtesting engine must not crash on data with gaps or on flat-line data. It must return a clear error message if the provided data is too short to calculate the required indicators.

### NFR6: Performance

- The core backtesting logic MUST be JIT-compiled using Numba to ensure high-speed execution
- Historical data fetches MUST be cached in memory to avoid redundant API calls during the same session

## 8. Deployment & Operations

### DEV1: Local Development & Validation

#### DEV1.1 (Running the Application)

The project must be fully runnable via a single `docker-compose up --build` command. The docker-compose.yml file should include a DNS fix (e.g., `dns: [1.1.1.1]`) to mitigate potential local network issues.

#### DEV1.2 (Running the Tests)

The unit test suite must be executable via `pytest -v`.

### PROD1 (Production Target - Render)

The system is designed for a two-service deployment on Render:

- A Web Service running the Streamlit UI
- A Background Worker running the alerter script

### PROD2 (Persistent State on Render)

A Render Persistent Disk MUST be used for alerts.json and state.json.
