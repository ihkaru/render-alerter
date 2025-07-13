import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import time
import ccxt
from datetime import datetime
from backtester import run_backtest

# --- Page Config and Helper Functions (No Changes) ---
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

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
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        return sorted([m for m in markets if m.endswith('/USDT') and not markets[m].get('future')])
    except Exception as e:
        st.error(f"Could not fetch asset list: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

def create_price_chart(price_data: pd.DataFrame, trades: pd.DataFrame, selected_timezone: str):
    if price_data.empty: return go.Figure()
    price_data.index = price_data.index.tz_convert(selected_timezone)
    if not trades.empty:
        trades['entry_ts'] = pd.to_datetime(trades['entry_ts']).dt.tz_convert(selected_timezone)
        trades['exit_ts'] = pd.to_datetime(trades['exit_ts']).dt.tz_convert(selected_timezone)
    fig = go.Figure(data=[go.Candlestick(x=price_data.index, open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close'], name='Price')])
    if not trades.empty:
        fig.add_trace(go.Scatter(x=trades['entry_ts'], y=trades['entry_price'], mode='markers', marker=dict(symbol='triangle-up', color='cyan', size=10), name='Buy'))
        fig.add_trace(go.Scatter(x=trades['exit_ts'], y=trades['exit_price'], mode='markers', marker=dict(symbol='triangle-down', color='yellow', size=10), name='Sell'))
    fig.update_layout(title='Price Chart with Trades', template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

def create_equity_chart(equity_curve: pd.Series, bnh_equity: pd.Series, drawdown_series: pd.Series, show_bnh: bool, selected_timezone: str):
    equity_curve.index = pd.to_datetime(equity_curve.index).tz_convert(selected_timezone)
    bnh_equity.index = pd.to_datetime(bnh_equity.index).tz_convert(selected_timezone)
    drawdown_series.index = pd.to_datetime(drawdown_series.index).tz_convert(selected_timezone)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Strategy Equity', line=dict(color='cyan', width=2)))
    if show_bnh:
        fig.add_trace(go.Scatter(x=bnh_equity.index, y=bnh_equity, mode='lines', name='Buy & Hold Equity', line=dict(color='orange', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, fill='tozeroy', mode='none', name='Drawdown', yaxis='y2', fillcolor='rgba(255, 82, 82, 0.3)'))
    fig.update_layout(title_text='Portfolio Equity Curve & Drawdown', template='plotly_dark', yaxis=dict(title='Portfolio Value ($)'), yaxis2=dict(title='Drawdown ($)', overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Main App ---
st.title("📈 Trading Strategy Backtester & Alert Manager")
config = load_json('config.json')
alerts = load_json('alerts.json', default_value=[])
if 'params' not in st.session_state:
    default_params = config.get('default_strategy_params', {})
    if not default_params:
        st.error("FATAL: `default_strategy_params` missing in config.json.")
        st.stop()
    st.session_state.params = default_params.copy()

col1, col2 = st.columns([1, 1])
symbol_to_use = "BTC/USDT" 

with col1:
    st.header("1. Configure & Backtest")
    with st.form(key="create_alert_form"):
        # --- Form widgets are unchanged ---
        st.subheader("Asset & Timeframe")
        asset_type=st.radio("Asset Type",["Crypto","Stocks"],horizontal=True,key='asset_type')
        if asset_type == "Crypto":
            asset_list = get_crypto_assets()
            default_index = asset_list.index("BTC/USDT") if "BTC/USDT" in asset_list else 0
            selected_symbol = st.selectbox("Select Asset", asset_list, index=default_index, key='symbol')
            symbol_to_use = selected_symbol
        else:
            asset_list = ["NVDA", "TSLA", "AAPL", "GOOGL", "MSFT"]
            selected_symbol_stock = st.selectbox("Select Asset (Stock)", asset_list, key='symbol_stock')
            symbol_to_use = selected_symbol_stock
        timeframes = ['1m','5m','15m','30m','1h','4h','1d','1w']
        selected_timeframe = st.selectbox("Select Timeframe", timeframes, index=3, key='timeframe')
        start_date_input = st.date_input("Backtest Start Date", value=datetime(2025, 1, 1), key='start_date')
        st.subheader("Strategy Parameters")
        p = st.session_state.params
        with st.expander("Show/Hide Full Strategy Parameters"):
            # All the parameter widgets remain the same
            st.markdown("##### Parameter Strategi Utama")
            p['ma_type'] = st.selectbox("Tipe Moving Average", ["SMA", "EMA"], index=["SMA", "EMA"].index(p['ma_type']), key='ma_type')
            p['short_window'] = st.number_input("Periode MA Cepat", value=p['short_window'], min_value=1, key='short_window')
            p['long_window'] = st.number_input("Periode MA Lambat", value=p['long_window'], min_value=1, key='long_window')
            st.divider()
            st.markdown("##### Parameter Indikator & Filter")
            p['use_price_ma_filter'] = st.checkbox("Gunakan Filter Harga di Bawah MA?", value=p.get('use_price_ma_filter', True), key='use_price_ma_filter')
            if p['use_price_ma_filter']:
                c1,c2 = st.columns(2)
                p['price_ma_filter_period'] = c1.number_input("Periode MA Filter", value=p['price_ma_filter_period'], key='price_ma_filter_period')
                p['price_ma_filter_type'] = c2.selectbox("Tipe MA Filter", ["SMA", "EMA"], index=["SMA", "EMA"].index(p['price_ma_filter_type']), key='price_ma_filter_type')
            p['use_rsi_filter'] = st.checkbox("Gunakan Filter RSI?", value=p.get('use_rsi_filter', True), key='use_rsi_filter')
            if p['use_rsi_filter']:
                p['rsi_period'] = st.number_input("Periode RSI", value=p['rsi_period'], key='rsi_period')
                c1,c2 = st.columns(2)
                p['rsi_buy_threshold'] = c1.number_input("RSI Beli <", value=p['rsi_buy_threshold'], key='rsi_buy_thresh')
                p['rsi_sell_threshold'] = c2.number_input("RSI Jual >", value=p['rsi_sell_threshold'], key='rsi_sell_thresh')
            # ... all other parameter widgets are the same
            p['use_volume_filter'] = st.checkbox("Gunakan Filter Volume?", value=p.get('use_volume_filter', False), key='use_volume_filter')
            if p['use_volume_filter']: p['volume_ma_period'] = st.number_input("Periode Volume MA", value=p['volume_ma_period'], key='volume_ma_period')
            p['use_volatility_filter'] = st.checkbox("Gunakan Filter Volatilitas (ATR)?", value=p.get('use_volatility_filter', False), key='use_volatility_filter')
            if p['use_volatility_filter']:
                p['atr_period'] = st.number_input("Periode ATR", value=p['atr_period'], key='atr_period')
                p['atr_min_value'] = st.number_input("Nilai ATR Minimum", value=p['atr_min_value'], format="%.2f", key='atr_min_value')
            p['use_spread_filter'] = st.checkbox("Gunakan Filter Spread MA?", value=p.get('use_spread_filter', False), key='use_spread_filter')
            if p['use_spread_filter']: p['spread_threshold'] = st.number_input("Ambang Batas Spread", value=p['spread_threshold'], format="%.4f", key='spread_threshold')
            p['use_price_dist_filter'] = st.checkbox("Gunakan Filter Jarak Harga ke MA?", value=p.get('use_price_dist_filter', True), key='use_price_dist_filter')
            if p['use_price_dist_filter']: p['price_dist_threshold'] = st.number_input("Ambang Batas Jarak (%)", value=p['price_dist_threshold'], format="%.4f", help="Value is a direct percentage, e.g., 0.2 means 0.2%", key='price_dist_threshold')
            p['use_cooldown_filter'] = st.checkbox("Gunakan Filter Cooldown (Jeda Beli)?", value=p.get('use_cooldown_filter', False), key='use_cooldown_filter')
            if p['use_cooldown_filter']: p['cooldown_bars'] = st.number_input("Jumlah Bar Cooldown", value=p['cooldown_bars'], key='cooldown_bars')
            p['use_hold_duration_filter'] = st.checkbox("Gunakan Filter Durasi Hold?", value=p.get('use_hold_duration_filter', False), key='use_hold_duration_filter')
            if p['use_hold_duration_filter']: p['min_hold_bars'] = st.number_input("Jumlah Bar Minimum Hold", value=p['min_hold_bars'], key='min_hold_bars')
            st.divider()
            st.markdown("##### Filter Slope untuk Sinyal Beli")
            slope_options = ["Nonaktif", "Gunakan MA Strategi", "Gunakan MA Kustom"]
            p['slope_filter_mode'] = st.selectbox("Mode Filter Slope", slope_options, index=slope_options.index(p.get('slope_filter_mode', 'Nonaktif')), key='slope_filter_mode')
            if p['slope_filter_mode'] != 'Nonaktif':
                if p['slope_filter_mode'] == "Gunakan MA Strategi":
                     ma_source_options = ["MA Cepat", "MA Lambat"]
                     p['strategy_ma_source'] = st.selectbox("   ↳ Sumber MA Strategi", ma_source_options, index=ma_source_options.index(p.get('strategy_ma_source', 'MA Cepat')), key='strategy_ma_source')
                if p['slope_filter_mode'] == "Gunakan MA Kustom":
                    p['custom_slope_ma_period'] = st.number_input("   ↳ Periode MA Kustom", value=p['custom_slope_ma_period'], key='custom_slope_ma_period')
                    custom_ma_type_options = ["SMA", "EMA"]
                    p['custom_slope_ma_type'] = st.selectbox("   ↳ Tipe MA Kustom", custom_ma_type_options, index=custom_ma_type_options.index(p.get('custom_slope_ma_type', 'EMA')), key='custom_slope_ma_type')
                p['slope_threshold'] = st.number_input("   ↳ Ambang Batas Slope Minimum", value=p['slope_threshold'], format="%.2f", key='slope_threshold')
            st.divider()
            st.markdown("##### Manajemen Stop Loss & Exit")
            p['use_stop_loss'] = st.checkbox("Gunakan Stop Loss?", value=p.get('use_stop_loss', True), key='use_stop_loss')
            if p['use_stop_loss']: p['stop_loss_pct'] = st.number_input("Stop Loss (%)", value=p['stop_loss_pct'], format="%.2f", key='sl_pct')
            p['use_recovery_exit'] = st.checkbox("Gunakan Recovery Exit?", value=p.get('use_recovery_exit', True), key='use_recovery_exit')
            if p['use_recovery_exit']:
                p['dip_pct_trigger'] = st.number_input("   ↳ Pemicu Penurunan Dalam (%)", value=p['dip_pct_trigger'], key='dip_trigger')
                p['recovery_pct_threshold'] = st.number_input("   ↳ Ambang Batas Pemulihan (%)", value=p['recovery_pct_threshold'], key='recovery_thresh')
            st.divider()
            st.markdown("##### Manajemen Risiko & Ukuran Posisi")
            p['base_position_pct'] = st.number_input("Persentase Modal Dasar (%)", value=p['base_position_pct'], key='base_pos_pct')
            p['leverage_multiplier'] = st.number_input("Leverage", value=p['leverage_multiplier'], min_value=1.0, key='leverage')
            p['use_size_down'] = st.checkbox("Aktifkan Aturan 'Size Down'?", value=p.get('use_size_down', True), key='use_size_down')
            if p['use_size_down']: p['size_down_pct'] = st.number_input("   ↳ Kurangi Ukuran Sebesar (%)", value=p['size_down_pct'], key='size_down_pct')
            p['use_fake_loss'] = st.checkbox("Aktifkan Aturan 'Fake Loss'?", value=p.get('use_fake_loss', True), key='use_fake_loss')
            if p['use_fake_loss']: p['withdrawal_pct'] = st.number_input("   ↳ Persentase 'Penarikan' (%)", value=p['withdrawal_pct'], key='withdrawal_pct')
        form_col1, form_col2 = st.columns(2)
        run_button = form_col1.form_submit_button("🚀 Run Backtest", use_container_width=True)
        add_alert_button = form_col2.form_submit_button("✅ Add as Live Alert", use_container_width=True)
# ---> THIS IS THE FIX: Logic is now sequential and simpler <---
# 1. Handle alert creation FIRST.
if add_alert_button:
    new_alert = {
        "id": f"{symbol_to_use.replace('/', '')}_{selected_timeframe}_{int(time.time())}",
        "symbol": symbol_to_use,
        "timeframe": selected_timeframe,
        "asset_type": asset_type.lower(),
        "params": st.session_state.params.copy()
    }
    alerts.append(new_alert)
    save_json('alerts.json', alerts)
    st.success(f"Alert '{new_alert['id']}' added successfully!")
    # We can still rerun to make the new alert appear instantly, it's safe now.
    time.sleep(1) 
    st.rerun()
# 2. Display the active alerts panel.
with col2:
    st.header("2. Active Alerts")
    if not alerts:
        st.info("No active alerts. Create one using the form on the left.")
    else:
        # Use a copy for safe iteration while removing items
        for i, alert in enumerate(alerts[:]):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**ID:** `{alert['id']}`")
                c1.markdown(f"**Asset:** {alert['symbol']} ({alert['timeframe']})")
                if c2.button("❌ Remove", key=f"remove_{alert['id']}", use_container_width=True):
                    alerts.pop(i)
                    save_json('alerts.json', alerts)
                    st.success(f"Alert '{alert['id']}' removed.")
                    st.rerun()

# 3. Handle backtest execution and display LAST.
st.header("3. Backtest Results")

if 'last_run_params' not in st.session_state: st.session_state.last_run_params = None
if run_button:
    live_config = {
        "strategy_params": st.session_state.params,
        "backtest_settings": config.get('backtest_settings', {}),
        "strategy_start_timestamp": pd.to_datetime(start_date_input).isoformat() + "Z"
    }
    with st.spinner(f"Running backtest for {symbol_to_use}..."):
        results = run_backtest(symbol_to_use, asset_type.lower(), selected_timeframe, live_config)

    # All results display logic is now nested inside the `if run_button:`
    if "error" in results:
        st.error(results["error"])
    elif "message" in results:
        st.info(results["message"])
        if "debug_info" in results:
            debug_info = results['debug_info']
            failed_counts = debug_info.get('failed_filter_counts', {})
            st.subheader("🕵️ Debugging: Why No Trades?")
            st.write(f"The backtester found **{debug_info.get('crossover_events_found', 0)}** potential buy signals (Golden Crosses), but they were all rejected by the active filters.")
            if failed_counts:
                st.write("Summary of which filters caused the rejections:")
                failed_df = pd.DataFrame(list(failed_counts.items()), columns=['Filter That Failed', 'Rejection Count'])
                failed_df = failed_df.sort_values(by='Rejection Count', ascending=False).reset_index(drop=True)
                st.dataframe(failed_df, use_container_width=True)
                
                # ---> NEW: Display Detailed Log <---
                detailed_log = debug_info.get('detailed_log', [])
                if detailed_log:
                    with st.expander("View Detailed Failure Log (First 20 Events)"):
                        log_df = pd.DataFrame(detailed_log)
                        st.dataframe(log_df, use_container_width=True)
                        st.warning("The most common reason for filter failures is the indicator value being right on the threshold (e.g., RSI is 50.1 instead of <50) or an indicator being invalid (NaN) at the start of the test. This new backtester version fixes the NaN issue.")
    else:
        # --- Display KPIs ---
        metrics = results.get('metrics', {})
        if metrics:
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Net Profit (%)", f"{metrics.get('net_profit_pct', 0):.2f}%")
            kpi_cols[1].metric("Total Trades", metrics.get('total_trades', 0))
            kpi_cols[2].metric("Win Rate (%)", f"{metrics.get('win_rate', 0):.2f}%")
            kpi_cols[3].metric("Max Drawdown", f"${metrics.get('max_drawdown', 0):,.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)")
            kpi_cols[4].metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        
        # --- Display Charts ---
        if 'historical_data' in results and 'trades' in results:
            st.subheader("Trade Visualization")
            df = pd.DataFrame.from_dict(results['historical_data'], orient='index')
            df.index = pd.to_datetime(df.index)
            st.plotly_chart(create_price_chart(df, pd.DataFrame(results['trades']), "Asia/Jakarta"), use_container_width=True)
        if 'equity_curve' in results:
            st.subheader("Equity Curve Analysis")
            show_bnh = st.checkbox("Show Buy & Hold Equity Comparison", value=True)
            equity_df = pd.Series(results['equity_curve'])
            bnh_df = pd.Series(results['buy_and_hold_equity'])
            drawdown_df = pd.Series(results['drawdown_series'])
            st.plotly_chart(create_equity_chart(equity_df, bnh_df, drawdown_df, show_bnh, "Asia/Jakarta"), use_container_width=True)
        
        # --- Display Trade Log ---
        with st.expander("View Trade Log"):
            if 'trades' in results and results['trades']:
                st.dataframe(pd.DataFrame(results['trades']), use_container_width=True)
else:
    # This message now shows by default if no backtest has been run
    st.info("Configure a strategy and click 'Run Backtest' to see results.")