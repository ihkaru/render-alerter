# System Requirements Document: Personal Trading Alerter & Backtester

**Version:** 4.0
**Date:** July 12, 2025
**Author:** Gemini & User
**Revision Note:** This version incorporates significant clarifications and logical refinements discovered during a comprehensive unit testing and debugging cycle. It represents the most accurate and validated system specification.

## 1. Project Vision & Purpose

This document outlines the system requirements for a personal, professional-grade quantitative trading analysis and alerting tool.

The primary purpose of this system is to provide its user with a reliable, non-repainting backtesting engine and a persistent live alerting mechanism. It is designed to be a fully-owned alternative to cloud platforms, giving the user complete control over the strategy logic, data sources, and operational environment. The overarching goal is to enable rapid strategy iteration and validation to support the user's investment portfolio growth.

## 2. Core Philosophy & Guiding Principles

The system is built upon several core principles that must be respected in all future development:

- **Non-Repainting Logic:** All backtesting and alerting logic MUST avoid look-ahead bias. Decisions are made on the data of a fully closed "signal candle," and actions (trades, alerts) are executed on the open of the subsequent "action candle."

- **(New & Clarified) Independent Exit Triggers vs. Filtered Entry:** The system's logic for entering and exiting positions is fundamentally different and MUST be respected.

  - **Entry Logic (AND):** A trade is only entered if the primary signal (e.g., a Golden Cross) occurs **AND** it passes a sequential chain of all enabled buy-side filters (e.g., RSI, Volume, Slope, etc.). A failure at any step prevents entry.
  - **Exit Logic (OR):** A trade is exited if **ANY** enabled exit condition is met. The conditions (Stop Loss, Recovery Exit, MA Slope-based Signal, RSI-based Signal) are evaluated independently. This ensures a protective stop loss, for example, is not blocked by another exit rule.

- **(New) End-of-Period Position Closure:** At the conclusion of a backtest, any open position MUST be automatically closed at the `close` price of the final available candle to ensure final equity is accurately reflected. This is a non-negotiable backstop to correctly calculate final P/L.

- **State Persistence:** The system must be resilient to restarts. The state of the live alerter (active positions) and the list of user-defined alerts are persisted to disk (`state.json`, `alerts.json`) and reloaded on startup.

- **Configuration-Driven:** The system's behavior is primarily defined by user-editable JSON configuration files, not hardcoded values.

- **Testability and Validation:** The system MUST be accompanied by a comprehensive suite of automated unit tests. These tests serve as a safety net to prevent regressions and validate that the core logic behaves exactly as specified by these requirements.

## 3. System Architecture & Technology Stack

- **Language:** Python 3.10+
- **Core Libraries:** pandas, numpy, pandas-ta, ccxt, yfinance, streamlit, plotly, schedule, requests
- **Testing Stack:** pytest, pytest-mock
- **Containerization:** Docker & Docker Compose

## 4. Data Flow & State Management

- **Configuration:** `config.json`, `alerts.json`
- **State:** `state.json`
- **Backtesting Workflow:**
  1. UI reads `config.json`.
  2. User modifies parameters in the UI, creating a temporary `live_config` object.
  3. `live_config` is passed to the `backtester` module.
  4. The backtester fetches data, runs the simulation, and returns a results dictionary.
  5. **(Updated)** The results dictionary is returned regardless of whether trades were executed.
  6. The UI displays the results.
- **Alerting Workflow:** The `alerter.py` script runs as a persistent background process, looping through `alerts.json` and checking for signals against the persistent state in `state.json`.

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

## 6. User Interface (UI) Requirements

The application UI is a web-based dashboard built with Streamlit.

#### UR1: Layout

- **UR1.1 (Left Column - The Control Panel):** Contains a single form (`st.form`) for configuring a strategy, including asset/timeframe selection and all strategy parameters. It has two action buttons: "Run Backtest" and "Add as Live Alert."
- **UR1.2 (Right Column - The Status Panel):** Displays the list of currently active alerts from `alerts.json`. Each alert has a "Remove" button.

#### UR2: Backtest Results

- **UR2.1 (KPIs):** Key Performance Indicators MUST be displayed prominently, including: Net Profit (%), Total Trades, Win Rate (%), Profit Factor, and **Max Drawdown ($ and %)**.
- **UR2.2 (Interactive Price Chart):** A Plotly candlestick chart must display the price action for the backtested period, with visual markers for all buy and sell trades.
- **UR2.3 (Equity Analysis Chart):** A dedicated, interactive Plotly chart for the portfolio equity must display:
  - `UR2.3.1:` The strategy's equity curve as a primary line graph.
  - `UR2.3.2:` The "Buy and Hold" equity curve as a dashed line. This MUST be toggleable with a checkbox on the UI.
  - `UR2.3.3:` The portfolio's drawdown over time, visualized as a secondary area graph on the same chart.
- **UR2.4: (Updated) Debugging & No-Trade Display:** If a backtest produces no trades, the UI MUST display the debugging information from F1.6. It should STILL render the price chart (UR2.2) using the `historical_data` that is always returned.
- **UR2.5 (Trade Log):** A collapsible, detailed table of every trade executed during the backtest must be available. **(Updated)** The underlying data for this log now contains the `entry_size_usd` for potential display.

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
