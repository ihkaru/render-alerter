import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # <-- MODIFICATION: Added import
import time
import ccxt
from datetime import datetime
from backtester import run_backtest
from optimizer import run_optimization

st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# --- Helper Functions (Unchanged) ---
def load_json(filepath: str, default_value=None):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_value if default_value is not None else {}

def save_json(filepath: str, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

@st.cache_data(ttl=3600)
def get_crypto_assets():
    try:
        exchange = ccxt.binanceus()
        markets = exchange.load_markets()
        return sorted([m for m in markets if m.endswith('/USDT') and not markets[m].get('future')])
    except Exception as e:
        st.error(f"Could not fetch asset list: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# --- MODIFICATION: Replaced the old chart function with a new combined one ---
def create_price_and_rsi_chart(price_data: pd.DataFrame, trades: pd.DataFrame, params: dict, selected_timezone: str):
    """
    Creates an interactive Plotly chart with the price/trades on the top subplot
    and the RSI on the bottom subplot, sharing the same x-axis.
    """
    if price_data.empty:
        return go.Figure()

    # Convert all relevant timestamps to the user's selected timezone
    price_data.index = price_data.index.tz_convert(selected_timezone)
    if not trades.empty:
        trades['entry_ts'] = pd.to_datetime(trades['entry_ts']).dt.tz_convert(selected_timezone)
        trades['exit_ts'] = pd.to_datetime(trades['exit_ts']).dt.tz_convert(selected_timezone)

    # Create a figure with 2 subplots that share the x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Price Chart", "Relative Strength Index (RSI)")
    )

    # --- Subplot 1: Price Candlestick Chart ---
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name='Price'
    ), row=1, col=1)

    # Add Moving Averages to the price chart
    if 'ma_short' in price_data.columns and price_data['ma_short'].notna().any():
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data['ma_short'], mode='lines',
            name=f"MA Cepat ({params.get('short_window')})",
            line=dict(color='#3498db', width=1.5)
        ), row=1, col=1)

    if 'ma_long' in price_data.columns and price_data['ma_long'].notna().any():
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data['ma_long'], mode='lines',
            name=f"MA Lambat ({params.get('long_window')})",
            line=dict(color='#f1c40f', width=1.5)
        ), row=1, col=1)

    # Add Buy/Sell trade markers to the price chart
    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades['entry_ts'], y=trades['entry_price'], mode='markers',
            marker=dict(symbol='triangle-up', color='cyan', size=12, line=dict(width=1, color='black')),
            name='Buy'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades['exit_ts'], y=trades['exit_price'], mode='markers',
            marker=dict(symbol='triangle-down', color='yellow', size=12, line=dict(width=1, color='black')),
            name='Sell'
        ), row=1, col=1)

    # --- Subplot 2: RSI Chart ---
    if 'rsi' in price_data.columns and price_data['rsi'].notna().any():
        rsi_period = params.get('rsi_period', 14)
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data['rsi'], mode='lines',
            name=f'RSI ({rsi_period})',
            line=dict(color='orange', width=1.5)
        ), row=2, col=1)

        if params.get('use_rsi_filter'):
            buy_thresh = params.get('rsi_buy_threshold')
            sell_thresh = params.get('rsi_sell_threshold')
            fig.add_hline(y=buy_thresh, line_dash="dash", line_color="lime", row=2, col=1,
                          annotation_text=f"Beli ({buy_thresh})", annotation_position="bottom right")
            fig.add_hline(y=sell_thresh, line_dash="dash", line_color="red", row=2, col=1,
                          annotation_text=f"Jual ({sell_thresh})", annotation_position="top right")

    # --- Layout Updates for the Combined Figure ---
    fig.update_layout(
        template='plotly_dark',
        height=750,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI Value", row=2, col=1, range=[0, 100])
    fig.update_xaxes(showticklabels=True)

    return fig


def create_equity_chart(equity_curve: pd.Series, bnh_equity: pd.Series, drawdown_series: pd.Series, show_bnh: bool, selected_timezone: str):
    equity_curve.index = pd.to_datetime(equity_curve.index).tz_convert(selected_timezone)
    bnh_equity.index = pd.to_datetime(bnh_equity.index).tz_convert(selected_timezone)
    drawdown_series.index = pd.to_datetime(drawdown_series.index).tz_convert(selected_timezone)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Strategy Equity'))
    if show_bnh:
        fig.add_trace(go.Scatter(x=bnh_equity.index, y=bnh_equity, mode='lines', name='Buy & Hold Equity', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, fill='tozeroy', mode='none', name='Drawdown', yaxis='y2'))
    fig.update_layout(title_text='Portfolio Equity Curve & Drawdown', template='plotly_dark', yaxis2=dict(overlaying='y', side='right'))
    return fig

st.title(":chart_with_upwards_trend: Trading Strategy Dashboard")
config = load_json('config.json')
alerts = load_json('alerts.json', default_value=[])

def initialize_state():
    if 'params_initialized' not in st.session_state:
        default_params = config.get('default_strategy_params', {})
        st.session_state.update(default_params)
        st.session_state.params_initialized = True
        st.session_state.start_date = datetime(2023, 1, 1)
        st.session_state.end_date = datetime.now()

def apply_loaded_params():
    if 'params_to_load' in st.session_state:
        params = st.session_state.params_to_load
        for key, value in params.items():
            if key in st.session_state:
                if isinstance(st.session_state[key], bool) and not isinstance(value, bool):
                     st.session_state[key] = bool(value)
                else:
                     st.session_state[key] = value
        del st.session_state['params_to_load']

initialize_state()
apply_loaded_params()

with st.sidebar:
    st.header("Strategy Controls")
    with st.expander("Ã°Å¸Å¡â‚¬ Run Parameter Optimizer"):
        st.info("To find the best parameters, use our powerful cloud-based optimizer on Google Colab.")
        st.markdown("1. [Click here to open the Optimizer Notebook](https://colab.research.google.com/).", unsafe_allow_html=True)
        st.markdown("2. Upload the `optimizer_colab.py` content to a new notebook.")
        st.markdown("3. Run the notebook, upload configs, and copy the final JSON output.")
        st.markdown("4. Paste the JSON into the 'Load Optimized Parameters' section in the main panel.")
    with st.expander("Asset & Timeframe", expanded=True):
        asset_type = st.radio("Asset Type", ["Crypto", "Stocks"], key='asset_type', horizontal=True)
        if st.session_state.asset_type == "Crypto":
            asset_list = get_crypto_assets()
            st.selectbox("Select Asset", asset_list, key='symbol')
        else:
            asset_list = ["NVDA", "TSLA", "AAPL", "GOOGL", "MSFT"]
            st.selectbox("Select Asset (Stock)", asset_list, key='symbol_stock')
        st.selectbox("Select Timeframe", ['1m','5m','15m','30m','1h','4h','1d','1w'], key='timeframe')
        st.date_input("Backtest Start Date", key='start_date')
        st.date_input("Backtest End Date", key='end_date')

    with st.expander("Strategy Parameters", expanded=False):
        st.markdown("##### Parameter Strategi Utama")
        st.selectbox("Tipe Moving Average", ["SMA", "EMA"], key='ma_type')
        st.number_input("Periode MA Cepat", min_value=1, key='short_window')
        st.number_input("Periode MA Lambat", min_value=1, key='long_window')
        st.divider()
        st.markdown("##### Parameter Indikator & Filter")
        st.checkbox("Gunakan Filter Harga di Bawah MA?", key='use_price_ma_filter')
        if st.session_state.get('use_price_ma_filter'):
            c1, c2 = st.columns(2)
            c1.number_input("Periode MA Filter", key='price_ma_filter_period')
            c2.selectbox("Tipe MA Filter", ["SMA", "EMA"], key='price_ma_filter_type')
        st.checkbox("Gunakan Filter RSI?", key='use_rsi_filter')
        if st.session_state.get('use_rsi_filter'):
            st.number_input("Periode RSI", key='rsi_period')
            c1, c2 = st.columns(2)
            c1.number_input("RSI Beli <", key='rsi_buy_threshold')
            c2.number_input("RSI Jual >", key='rsi_sell_threshold')
        st.checkbox("Gunakan Filter Volume?", key='use_volume_filter')
        if st.session_state.get('use_volume_filter'):
            st.number_input("Periode Volume MA", key='volume_ma_period')
        st.checkbox("Gunakan Filter Volatilitas (ATR)?", key='use_volatilidad_filter')
        if st.session_state.get('use_volatility_filter'):
            st.number_input("Periode ATR", key='atr_period')
            st.number_input("Nilai ATR Minimum", format="%.2f", key='atr_min_value')
        st.checkbox("Gunakan Filter Spread MA?", key='use_spread_filter')
        if st.session_state.get('use_spread_filter'):
            st.number_input("Ambang Batas Spread", format="%.4f", key='spread_threshold')
        st.checkbox("Gunakan Filter Jarak Harga ke MA?", key='use_price_dist_filter')
        if st.session_state.get('use_price_dist_filter'):
            st.number_input("Ambang Batas Jarak (%)", format="%.4f", help="Value is a direct percentage, e.g., 0.2 means 0.2%", key='price_dist_threshold')
        st.checkbox("Gunakan Filter Cooldown (Jeda Beli)?", key='use_cooldown_filter')
        if st.session_state.get('use_cooldown_filter'):
            st.number_input("Jumlah Bar Cooldown", key='cooldown_bars')
        st.checkbox("Gunakan Filter Durasi Hold?", key='use_hold_duration_filter')
        if st.session_state.get('use_hold_duration_filter'):
            st.number_input("Jumlah Bar Minimum Hold", key='min_hold_bars')
        st.divider()
        st.markdown("##### Manajemen Stop Loss & Exit")
        st.checkbox("Gunakan Stop Loss?", key='use_stop_loss')
        if st.session_state.get('use_stop_loss'):
            st.number_input("Stop Loss (%)", format="%.2f", key='stop_loss_pct')
        st.checkbox("Gunakan Recovery Exit?", key='use_recovery_exit')
        if st.session_state.get('use_recovery_exit'):
            st.number_input("   -> Pemicu Penurunan Dalam (%)", key='dip_pct_trigger')
            st.number_input("   -> Ambang Batas Pemulihan (%)", key='recovery_pct_threshold')
        st.divider()
        st.markdown("##### Manajemen Risiko & Ukuran Posisi")
        st.number_input("Persentase Modal Dasar (%)", key='base_position_pct')
        st.number_input("Leverage", min_value=1.0, key='leverage_multiplier')
        st.checkbox("Aktifkan Aturan 'Size Down'?", key='use_size_down')
        if st.session_state.get('use_size_down'):
            st.number_input("   -> Kurangi Ukuran Sebesar (%)", key='size_down_pct')
        st.checkbox("Aktifkan Aturan 'Fake Loss'?", key='use_fake_loss')
        if st.session_state.get('use_fake_loss'):
            st.number_input("   -> Persentase 'Penarikan' (%)", key='withdrawal_pct')

    st.header("Actions")
    run_button = st.button(":rocket: Run Backtest", use_container_width=True)
    add_alert_button = st.button(":white_check_mark: Add as Live Alert", use_container_width=True)
    st.divider()
    optimize_button = st.button(":mag: Find Optimized Parameters", use_container_width=True)

current_params = {key: st.session_state.get(key) for key in config.get('default_strategy_params', {}).keys()}
symbol_to_use = st.session_state.symbol if st.session_state.asset_type == 'Crypto' else st.session_state.symbol_stock
selected_timeframe = st.session_state.timeframe

# --- START OF FIX: Implemented the logic for the "Add as Live Alert" button ---
if add_alert_button:
    asset_type = st.session_state.asset_type.lower()
    
    # Create a unique ID to prevent conflicts.
    unique_id = f"{symbol_to_use.replace('/', '')}_{selected_timeframe}_{int(time.time())}"
    
    # Construct the new alert dictionary in the required format.
    new_alert = {
        "id": unique_id,
        "symbol": symbol_to_use,
        "timeframe": selected_timeframe,
        "asset_type": asset_type,
        "params": current_params
    }
    
    # Append the new alert to the list and save it to the JSON file.
    alerts.append(new_alert)
    save_json('alerts.json', alerts)
    
    # Provide user feedback and refresh the UI to show the new alert.
    st.success(f"Alert '{unique_id}' added successfully!")
    time.sleep(1) # Short delay for the user to read the message.
    st.rerun()
# --- END OF FIX ---

if 'last_run_results' not in st.session_state: st.session_state.last_run_results = None
if 'optimization_results' not in st.session_state: st.session_state.optimization_results = None

st.header("Ã¢Å¡Â¡Ã¯Â¸  Load Optimized Parameters from Colab")
colab_json_input = st.text_area(
    "Paste JSON output from Google Colab here:",
    height=200,
    key="colab_json",
    help="After running the optimizer in Colab, copy the final JSON block and paste it here."
)
if st.button("Load & Display Optimized Parameters"):
    if colab_json_input:
        try:
            parsed_data = json.loads(colab_json_input)
            st.session_state.optimization_results = pd.DataFrame(parsed_data)
            st.session_state.last_run_results = None
            st.success("Successfully loaded optimization results! See table below.")
            st.rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please copy the entire block from Colab, including `[` and `]`.")
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
    else:
        st.warning("Text area is empty. Please paste the JSON from Colab.")

st.divider()

if optimize_button:
    st.session_state.last_run_results = None
    with st.spinner(f"Running optimization..."):
        st.session_state.optimization_results = run_optimization(symbol_to_use, st.session_state.asset_type.lower(), {"strategy_params": current_params, "backtest_settings": config.get('backtest_settings', {})})

if run_button:
    st.session_state.optimization_results = None
    live_config = {"strategy_params": current_params, "backtest_settings": config.get('backtest_settings', {}), "strategy_start_timestamp": pd.to_datetime(st.session_state.start_date).isoformat() + "Z", "strategy_end_timestamp": pd.to_datetime(st.session_state.end_date).isoformat() + "Z"}
    with st.spinner(f"Running backtest..."):
        st.session_state.last_run_results = run_backtest(symbol_to_use, st.session_state.asset_type.lower(), selected_timeframe, live_config)
    st.session_state.last_run_config = live_config

results = st.session_state.get('last_run_results')
live_config = st.session_state.get('last_run_config')
optimization_results = st.session_state.get('optimization_results')

if optimization_results is not None:
    st.header("Top 5 Optimized Parameter Sets")
    st.info("Click 'Load' to apply a parameter set, then run a detailed backtest.")
    optimizer_config = load_json('optimizer_config.json', {})
    date_range_space = optimizer_config.get('date_range_space', {})
    edited_df = optimization_results.copy()
    edited_df["Load"] = False
    edited_df = st.data_editor(
        edited_df,
        column_config={"Load": st.column_config.CheckboxColumn(required=True)},
        disabled=[col for col in optimization_results.columns],
        hide_index=True,
    )
    if edited_df["Load"].any():
        selected_row = edited_df[edited_df["Load"]].iloc[0]
        params_to_load = {}
        for param, value in selected_row.items():
            if param == 'date_range_name':
                params_to_load['start_date'] = pd.to_datetime(date_range_space[value]['start_date'])
                params_to_load['end_date'] = pd.to_datetime(date_range_space[value]['end_date'])
            elif param != 'Load' and param != 'Score':
                 params_to_load[param] = value
        st.session_state.params_to_load = params_to_load
        st.session_state.optimization_results = None
        st.rerun()

elif results and live_config:
    st.header("Backtest Results")
    if "error" in results:
        st.error(results["error"])
        st.session_state.last_run_results = None
    else:
        if 'historical_data' in results:
            st.subheader("Trade Visualization")
            df = pd.DataFrame.from_dict(results['historical_data'], orient='index')
            df.index = pd.to_datetime(df.index)
            trades_df = pd.DataFrame(results.get('trades', []))
            # --- MODIFICATION: Call the new, combined charting function ---
            st.plotly_chart(create_price_and_rsi_chart(df, trades_df, live_config['strategy_params'], "Asia/Jakarta"), use_container_width=True)

        if 'metrics' in results and 'equity_curve' in results:
            metrics = results['metrics']
            st.subheader("Key Performance Indicators")
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Net Profit (%)", f"{metrics.get('net_profit_pct', 0):.2f}%")
            kpi_cols[1].metric("Total Trades", metrics.get('total_trades', 0))
            kpi_cols[2].metric("Win Rate (%)", f"{metrics.get('win_rate', 0):.2f}%")
            kpi_cols[3].metric("Max Drawdown", f"${metrics.get('max_drawdown', 0):,.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)")
            kpi_cols[4].metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            
            st.subheader("Equity Curve Analysis")
            show_bnh = st.checkbox("Show Buy & Hold Equity Comparison", value=True, key="show_bnh")
            equity_df = pd.Series(results['equity_curve'])
            bnh_df = pd.Series(results['buy_and_hold_equity'])
            drawdown_df = pd.Series(results['drawdown_series'])
            st.plotly_chart(create_equity_chart(equity_df, bnh_df, drawdown_df, show_bnh, "Asia/Jakarta"), use_container_width=True)
            
        if 'trades' in results and results.get('trades'):
             with st.expander("View Detailed Trade Log", expanded=True):
                st.dataframe(pd.DataFrame(results['trades']), use_container_width=True)
        elif "message" in results:
             st.info(results["message"])

else:
    st.info("Open the sidebar, configure a strategy, and click 'Run Backtest' or 'Find Optimized Parameters'.")

with st.expander("Show/Hide Active Alerts"):
    if not alerts:
        st.info("No active alerts. Add one from the sidebar.")
    else:
        for i, alert in enumerate(alerts[:]):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**ID:** `{alert['id']}`")
                c1.markdown(f"**Asset:** {alert['symbol']} ({alert['timeframe']})")
                if c2.button(":x: Remove", key=f"remove_{alert['id']}", use_container_width=True):
                    alerts.pop(i)
                    save_json('alerts.json', alerts)
                    st.success(f"Alert '{alert['id']}' removed.")
                    st.rerun()