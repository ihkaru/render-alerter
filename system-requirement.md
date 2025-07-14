# System Requirements Document: Personal Trading Alerter & Backtester

**Version:** 5.0
**Date:** July 13, 2025
**Author:** Gemini & User
**Revision Note:** This version introduces a powerful strategy optimization engine using Optuna for automated parameter discovery and robustness testing. The UI has been significantly refactored to a chart-centric, sidebar-based layout for improved usability, and includes on-chart indicator visualization.

## 1. Project Vision & Purpose

This document outlines the system requirements for a personal, professional-grade quantitative trading analysis and alerting tool.

The primary purpose of this system is to provide its user with a reliable, non-repainting backtesting engine and a persistent live alerting mechanism. It is designed to be a fully-owned alternative to cloud platforms, giving the user complete control over the strategy logic, data sources, and operational environment. The overarching goal is to enable rapid strategy iteration and validation to support the user's investment portfolio growth. Furthermore, it introduces an **automated optimization engine** to systematically discover robust strategy parameters, accelerating the validation process.

## 2. Core Philosophy & Guiding Principles

The system is built upon several core principles that must be respected in all future development:

- **Non-Repainting Logic:** All backtesting and alerting logic MUST avoid look-ahead bias. Decisions are made on the data of a fully closed "signal candle," and actions (trades, alerts) are executed on the open of the subsequent "action candle."
- **Independent Exit Triggers vs. Filtered Entry:** The system's logic for entering and exiting positions is fundamentally different and MUST be respected.
  - **Entry Logic (AND):** A trade is only entered if the primary signal (e.g., a Golden Cross) occurs **AND** it passes a sequential chain of all enabled buy-side filters.
  - **Exit Logic (OR):** A trade is exited if **ANY** enabled exit condition is met. The conditions are evaluated independently.
- **End-of-Period Position Closure:** At the conclusion of a backtest, any open position MUST be automatically closed at the `close` price of the final available candle.
- **State Persistence:** The system must be resilient to restarts. The state of the live alerter and the list of user-defined alerts are persisted to disk.
- **Configuration-Driven:** The system's behavior is primarily defined by user-editable JSON configuration files.
- **Testability and Validation:** The system MUST be accompanied by a comprehensive suite of automated unit tests.

## 3. System Architecture & Technology Stack

- **Language:** Python 3.10+
- **Core Libraries:** pandas, numpy, pandas-ta, ccxt, yfinance, streamlit, plotly, schedule, requests, **optuna**
- **Testing Stack:** pytest, pytest-mock
- **Containerization:** Docker & Docker Compose

## 4. Data Flow & State Management

- **Configuration:** `config.json`, `alerts.json`, **`optimizer_config.json`**
- **State:** `state.json`
- **Backtesting Workflow:** (Unchanged)
- **Alerting Workflow:** (Unchanged)
- **(New) Optimization Workflow:**
  1. The UI reads `optimizer_config.json` to define the search space for parameters, timeframes, and historical date ranges.
  2. The user initiates the optimization process from the UI.
  3. The `optimizer` module, using Optuna, runs a series of backtests (`N` trials).
  4. Each trial tests a unique combination of parameters and conditions (e.g., a specific timeframe over a specific historical market regime).
  5. The optimizer evaluates each trial based on a balanced score and returns a DataFrame of the top N best-performing parameter sets.
  6. The UI displays these results in a table, allowing the user to load any set of parameters back into the main controls for detailed analysis or live alerting.

## 5. Functional Requirements (Features)

#### F1: Strategy Backtesting Engine

- **F1.1:** Must perform a stateful, non-repainting backtest as per the logic defined in section 2.
- **F1.2:** Must support configurable commission costs, applied to both entry and exit of a trade.
- **F1.3: (Updated & Expanded) Comprehensive Risk & Exit Management:** The engine must correctly implement all of the following stateful rules based on the user's configuration. The exit conditions operate under **OR** logic.

  - **`Stop Loss`:** Must exit a trade at the exact stop-loss price if the `low` of any action candle touches or crosses below it.
  - **`Recovery Exit`:** Must correctly track a trade that has dipped significantly and exit it if the price recovers to near the entry point, as defined by the thresholds.
  - **`RSI Sell Threshold`:** Must trigger an exit if the RSI exceeds the defined sell threshold, independent of other signals.
  - **`Minimum Hold Duration`:** Must _ignore_ valid primary sell signals (like a negative MA slope) if the trade has not been open for the required number of bars.
  - **`Size Down`:** Must correctly reduce the position size for the next trade following a losing trade, based on the configured percentage.
  - **`Fake Loss`:** Must correctly calculate a separate `adjusted_equity` by "withdrawing" a percentage of profits, and use this reduced equity value for all subsequent position sizing calculations.
  - **`Leverage Multiplier`:** Must correctly scale the calculated `entry_size_usd` by the specified multiplier.
  - **`Cooldown Period`:** Must prevent a new buy signal from executing if it occurs within the configured number of bars after the previous trade's exit.

- **F1.4: Timeframe-Aware Indicator Warm-up:** Must fetch sufficient historical data and correctly align the backtest start to the first candle where all required indicators (based on the longest lookback period) are valid. The user's requested start date is a minimum; the actual start may be later.

- **F1.5: (Updated) Comprehensive Results Dictionary:** Must output a results dictionary containing performance metrics, a detailed trade log, and equity curve data. Crucially, it MUST **always** return the `historical_data` and `debug_info` objects, even if no trades were executed.

- **F1.6: Backtest Debugging Output:** If a backtest results in zero trades, the `debug_info` object MUST contain the count of primary buy signals and a breakdown of which filters caused the rejections.

- **F1.7: (New & Validated) End-of-Period Position Closure:** If a position is still open when historical data is exhausted, the backtester MUST automatically close the trade at the `close` price of the final candle.
- **(New) F1.8: Backtest over Specific Date Ranges:** The backtesting engine must accept an optional `end_date` to allow for analysis of specific historical periods (e.g., bull or bear markets).

#### F2: Live Alerting Service

- **F2.1:** Must run as a continuous, scheduled background process.
- **F2.2:** Must read its list of jobs from `alerts.json`.
- **F2.3:** (Updated) Must process each alert independently, implementing the same signal and filter logic as the backtester (including `Recovery Exit`, `RSI Sell`, etc.) for maximum consistency.
- **F2.4:** Must maintain a persistent state for each alert using `state.json`.
- **F2.5:** Must integrate with the Telegram API.

#### F3: Alert Management (CRUD)

- **F3.1 (Create):** The user must be able to create a new live alert from the UI. This action saves the currently configured symbol, timeframe, and strategy parameters to `alerts.json`.
- **F3.2 (Read):** The UI must display a list of all currently active alerts from `alerts.json`.
- **F3.3 (Delete):** The user must be able to remove an active alert via a "Remove" button in the UI.

#### F4: Data & Configuration

- **F4.1:** The UI must provide widgets for every single parameter present in the reference Pine Script strategy.
- **F4.2:** The UI must dynamically fetch and display a list of available crypto assets from the configured exchange.

#### F5: Performance Analytics

- **F5.1: Buy and Hold Comparison:** The backtester MUST calculate the full equity curve for a simple "Buy and Hold" strategy over the same period for comparison.
- **F5.2: Drawdown Calculation:** The backtester MUST calculate the portfolio's drawdown series and identify the maximum drawdown value ($) and percentage (%).

#### (New) F6: Strategy Optimization Engine

- **F6.1: Automated Parameter Search:** The system MUST use the Optuna library to intelligently search for optimal strategy parameters, efficiently exploring the defined search space.
- **F6.2: Configurable Search Space:** The search space for the optimizer—including parameter ranges, categorical choices (like MA type), timeframes, and named date ranges—MUST be defined in the `optimizer_config.json` file.
- **F6.3: Robustness Testing (Market Regimes):** The optimizer MUST be able to run trials across different, pre-defined historical date ranges (e.g., "Bull_Run_2021", "Bear_Market_2022") to find parameters that are robust across various market conditions.
- **F6.4: Multi-Objective Scoring:** The optimizer MUST evaluate each backtest trial using a single, balanced score metric that rewards profitability and stability while penalizing high drawdowns (e.g., `Score = (Profit Factor * Win Rate) / Max Drawdown %`).
- **F6.5: Top Results Display:** The optimizer MUST return the top 5 best-performing parameter sets, including the timeframe and date range tested, as a DataFrame for display in the UI.

## 6. User Interface (UI) Requirements

The application UI is a web-based dashboard built with Streamlit.

#### UR1: Layout

- **(Updated) UR1.1: Sidebar Control Panel:** All strategy configuration widgets (asset/timeframe selection, all strategy parameters) and primary action buttons ("Run Backtest", "Add as Live Alert", "Find Optimized Parameters") MUST be located in a collapsible `st.sidebar`.
- **(Updated) UR1.2: Main Display Area:** The main area of the application is dedicated to displaying the output of a user action, which is either the detailed backtest results or the strategy optimization results table.
- **(Updated) UR1.3: Collapsible Alerts Panel:** The list of currently active alerts from `alerts.json` MUST be displayed in a collapsible `st.expander` at the bottom of the main page.

#### UR2: Backtest Results

- **UR2.1 (KPIs):** Key Performance Indicators MUST be displayed prominently.
- **(Updated) UR2.2 (Interactive Price Chart):** A Plotly candlestick chart must display the price action, trade markers, **and key strategy indicators** (e.g., short and long-term moving averages) plotted directly on the price series.
- **UR2.3 (Equity Analysis Chart):** A dedicated, interactive Plotly chart must display the portfolio equity curve and drawdown.
- **UR2.4 (Debugging & No-Trade Display):** If a backtest produces no trades, the UI MUST still render the price chart and display the debugging information.
- **UR2.5 (Trade Log):** A collapsible, detailed table of every trade executed must be available.

#### (New) UR3: Optimization Results

- **UR3.1: Top 5 Results Table:** The UI must display the results from the optimizer (F6.5) in a clear, interactive table format (e.g., `st.data_editor`).
- **UR3.2: Interactive Parameter Loading:** Each row in the optimization results table MUST have a "Load" button. Clicking this button MUST instantly update all corresponding control widgets in the sidebar (UR1.1) to match the values from that row, including the `timeframe` and `start_date`/`end_date` widgets.

## 7. Non-Functional Requirements

- **NFR1 (Reliability):** The `alerter.py` service must be configured to restart automatically on failure.
- **NFR2 (Security):** All sensitive credentials MUST be managed via environment variables.
- **NFR3 (Usability):** The application must be fully runnable within a local Docker environment with clear instructions.
- **NFR4 (Testability & Validation):** The project MUST maintain a suite of unit tests using `pytest` to validate core logic and prevent regressions.
- **(New) NFR5: Robustness to Imperfect Data:** The backtesting engine must handle common real-world data issues gracefully. It must not crash on data with gaps (missing rows) or on data that is perfectly flat. It must return a clear error message if the provided data is too short to calculate the required indicators.

## 8. Deployment & Operations

- **(Updated) DEV1: Local Development & Validation:** The local development workflow is now twofold:
  - **DEV1.1 (Running the Application):** The project must be fully runnable via a single `docker-compose up --build` command after installing dependencies with `pip install -r requirements.txt`.
  - **DEV1.2 (Running the Tests):** The unit test suite must be executable via the `pytest -v` command after installing both application and development dependencies (`pip install -r requirements.txt -r requirements-dev.txt`).
- **PROD1 (Production Target - Render):** The system is designed for a two-service deployment on Render:
  - A **Web Service** running the Streamlit UI.
  - A **Background Worker** running the alerter script.
- **PROD2 (Persistent State on Render):** A Render Persistent Disk MUST be used for `alerts.json` and `state.json`.
