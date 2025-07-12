# System Requirements Document: Personal Trading Alerter & Backtester

**Version:** 2.0
**Date:** July 12, 2025
**Author:** Gemini & User

## 1. Project Vision & Purpose

This document outlines the system requirements for a personal, professional-grade quantitative trading analysis and alerting tool.

The primary purpose of this system is to provide its user with a reliable, non-repainting backtesting engine and a persistent live alerting mechanism. It is designed to be a fully-owned alternative to cloud platforms like TradingView, giving the user complete control over the strategy logic, data sources, and operational environment. The overarching goal is to enable rapid strategy iteration and validation to support the user's investment portfolio growth.

## 2. Core Philosophy & Guiding Principles

The system is built upon several core principles that must be respected in all future development:

- **Non-Repainting Logic:** All backtesting and alerting logic MUST avoid look-ahead bias. Decisions are made on the data of a fully closed "signal candle," and actions (trades, alerts) are executed on the open of the subsequent "action candle."
- **State Persistence:** The system must be resilient to restarts. The state of the live alerter (active positions) and the list of user-defined alerts are persisted to disk (`state.json`, `alerts.json`) and reloaded on startup.
- **Configuration-Driven:** The system's behavior is primarily defined by user-editable JSON configuration files, not hardcoded values. This allows for flexibility and easy management.
- **Modularity:** The codebase is separated into logical modules (shared_logic, backtester, alerter, app), each with a distinct responsibility. This promotes maintainability and testability.
- **Visual-First Analysis:** The primary output of a backtest is not just a table of numbers, but an interactive chart that allows for deep visual inspection of every trade, empowering the user to understand _why_ the strategy behaved as it did.

## 3. System Architecture & Technology Stack

The application is a containerized, multi-service system designed for both local development and cloud deployment.

- **Language:** Python 3.10+
- **Core Libraries:**
  - Data Analysis: pandas, numpy
  - Technical Analysis: pandas-ta
  - Data Fetching: ccxt (for crypto), yfinance (for stocks)
  - Web UI: streamlit
  - Charting: plotly
  - Scheduling: schedule
  - API Communication: requests
- **Containerization:** Docker & Docker Compose
- **Target Deployment Environment:** A Platform-as-a-Service (PaaS) like Render, or a self-managed VPS.

## 4. Data Flow & State Management

The system operates with a clear flow of data and state:

- **Configuration:**
  - `config.json`: Stores global settings and default strategy parameters.
  - `alerts.json`: Stores a list of user-created, persistent alert configurations. Each alert has a unique ID, symbol, timeframe, and a full set of its own strategy parameters.
- **State:**
  - `state.json`: A simple key-value store mapping `alert_id` to its current position status (e.g., `{"in_position": true, "entry_price": 50000}`). This file is the alerter's "memory."
- **Backtesting Workflow:**
  1.  The Streamlit UI reads the default parameters from `config.json` to populate the form.
  2.  The user modifies parameters in the UI.
  3.  Upon clicking "Run Backtest," the UI constructs a temporary `live_config` object from the current state of the widgets.
  4.  This `live_config` is passed to the `backtester` module.
  5.  The backtester fetches deep historical data, runs the non-repainting simulation, and returns a dictionary of results (metrics, trades, chart data, etc.).
  6.  The UI displays these results.
- **Alerting Workflow:**
  1.  The `alerter.py` script runs as a persistent background process.
  2.  On a schedule, it reads `alerts.json` and `state.json`.
  3.  It loops through each defined alert.
  4.  For each alert, it fetches recent market data, checks the strategy conditions (non-repainting), and compares against its state in `state.json`.
  5.  If conditions are met for a new signal, it sends a Telegram notification and updates `state.json`.

## 5. Functional Requirements (Features)

#### F1: Strategy Backtesting Engine

- **F1.1:** Must perform a stateful, non-repainting backtest as per the logic defined in section 2.
- **F1.2:** Must support configurable commission costs, applied to both entry and exit of a trade.
- **F1.3:** Must correctly implement all risk management rules defined in the Pine Script, including stateful equity tracking, "Size Down," and "Fake Loss" simulations.
- **F1.4:** Must implement a timeframe-aware indicator "warm-up" period, fetching sufficient historical data and correctly aligning the backtest start to ensure indicators are valid from the very first evaluated candle.
- **F1.5:** Must output a comprehensive results dictionary containing performance metrics, a detailed trade log, equity curve data, and all analytics data required by the UI.
- **F1.6:** **Backtest Debugging Output:** If a backtest results in zero trades, the engine MUST return a `debug_info` object containing:
  - The total count of primary buy signals (e.g., Golden Crosses) found.
  - A dictionary counting the number of times each specific filter (RSI, Price Distance, etc.) was the reason for a rejected trade.
  - A detailed log of the first ~20 failed events, showing the timestamp and the exact filter(s) that failed.

#### F2: Live Alerting Service

- **F2.1:** Must run as a continuous, scheduled background process.
- **F2.2:** Must read its list of jobs from `alerts.json`.
- **F2.3:** Must process each alert independently using its specific configuration.
- **F2.4:** Must maintain a persistent state for each alert using `state.json` to track open positions across restarts.
- **F2.5:** Must integrate with the Telegram API to send formatted alert messages. Secrets (Token, Chat ID) must be loaded from environment variables.

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
- **UR2.4 (Debugging Display):** If a backtest produces no trades, the UI must display the debugging information from `F1.6`, including the summary table of failed filters and the detailed log within an expander.
- **UR2.5 (Trade Log):** A collapsible, detailed table of every trade executed during the backtest must be available.

## 7. Non-Functional Requirements

- **NFR1 (Reliability):** The `alerter.py` service must be configured to restart automatically on failure (`restart: always` in Docker Compose).
- **NFR2 (Security):** All sensitive credentials (e.g., Telegram Token) MUST be managed via environment variables. The `.env` file and state files (`state.json`, `alerts.json`) MUST be included in `.gitignore` and never committed to version control.
- **NFR3 (Usability):** The application must be fully runnable within a local Docker environment with clear instructions.

## 8. Deployment & Operations

- **DEV1 (Local Development):** The project must be fully runnable via a single `docker-compose up --build` command.
- **PROD1 (Production Target - Render):** The system is designed for a two-service deployment on Render:
  - A **Web Service** running the Streamlit UI. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - A **Background Worker** running the alerter script. Start command: `python alerter.py`
- **PROD2 (Persistent State on Render):** A Render **Persistent Disk** MUST be created and attached to **BOTH** the Web Service and the Background Worker. The mount path for this disk MUST be `/app` to ensure both services read from and write to the same `alerts.json` and `state.json` files.
