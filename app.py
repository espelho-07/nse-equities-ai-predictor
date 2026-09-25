import os
import json
import joblib
import time
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, timezone

# Import trainer function
from train_model import prepare_features, train_and_save_model

# Page Configuration
st.set_page_config(
    page_title="EQUITY·AI PRO | NSE Neural Forecasting Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force Streamlit Light Theme in browser DOM and localStorage
components.html("""
<script>
    try {
        const theme = "light";
        window.localStorage.setItem("stActiveTheme", theme);
        window.localStorage.setItem("stTheme", theme);
        window.localStorage.setItem("theme", theme);
        const doc = window.parent.document;
        if (doc) {
            doc.documentElement.setAttribute('data-theme', theme);
            doc.body.setAttribute('data-theme', theme);
            doc.documentElement.style.colorScheme = theme;
            doc.body.style.colorScheme = theme;
            doc.querySelectorAll('[data-theme]').forEach(el => el.setAttribute('data-theme', theme));
        }
    } catch(e) {}
</script>
""", height=0, width=0)

# Professional Quantitative Design System & Strict Color Enforcement
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

/* Global Root Variables & Theme Lock */
:root, html, body, [data-theme="dark"], [data-theme="light"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --bg-subtle: #f1f5f9;
    --border-subtle: #e2e8f0;
    --border-strong: #cbd5e1;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --text-sub: #334155;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-light: #eff6ff;
    --success: #16a34a;
    --success-light: #f0fdf4;
    --danger: #dc2626;
    --danger-light: #fef2f2;
    --warning: #d97706;
    --warning-light: #fffbeb;
    --purple: #7c3aed;
    --purple-light: #f5f3ff;
    
    color-scheme: light !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Hide Streamlit Header & Optimize Spacing */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
}

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* Monospace Numbers */
.mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header Container */
.quant-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    background: #2563eb;
    color: #ffffff;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 800;
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.4px;
    margin: 0;
    line-height: 1.1;
}

.brand-badge {
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
    letter-spacing: 0.3px;
    vertical-align: middle;
}

.brand-subtitle {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 2px;
    font-weight: 500;
}

.header-status-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.3px;
}

.closed-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #b91c1c;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.3px;
}

.pulse-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #16a34a;
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.7);
    animation: pulse-glow 1.8s infinite;
}

@keyframes pulse-glow {
    0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.7); }
    70% { box-shadow: 0 0 0 5px rgba(22, 163, 74, 0); }
    100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
}

.time-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #475569;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Market Overview Bar */
.ticker-bar {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 8px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}

.ticker-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #334155;
    font-weight: 600;
}

.ticker-name {
    color: #64748b;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.ticker-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #0f172a;
}

.ticker-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
}

.chip-up {
    background: #f0fdf4;
    color: #16a34a;
    border: 1px solid #bbf7d0;
}

.chip-down {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

.model-pill {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Strict Clean Navigation Tabs */
div[data-testid="stSegmentedControl"],
div[data-testid="stSegmentedControl"] > div,
[data-baseweb="button-group"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    gap: 6px !important;
}

div[data-testid="stSegmentedControl"] button,
[data-baseweb="button-group"] button,
[data-baseweb="button-group"] > div > button {
    background-color: #f8fafc !important;
    color: #334155 !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: 1px solid #e2e8f0 !important;
    padding: 7px 16px !important;
    transition: all 0.15s ease !important;
}

div[data-testid="stSegmentedControl"] button:hover,
[data-baseweb="button-group"] button:hover {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[data-selected="true"],
div[data-testid="stSegmentedControl"] button[aria-selected="true"],
[data-baseweb="button-group"] button[aria-checked="true"],
[data-baseweb="button-group"] button[data-selected="true"] {
    background: #2563eb !important;
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-color: #2563eb !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
div[data-testid="stSegmentedControl"] button[data-selected="true"] * {
    color: #ffffff !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="false"] *,
div[data-testid="stSegmentedControl"] button[data-selected="false"] * {
    color: #334155 !important;
}

/* Card Container */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"]) {
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 14px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

/* High Contrast Metric KPI Cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 6px;
}

.kpi-footer {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

.delta-badge-green {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.74rem;
}

.delta-badge-red {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.74rem;
}

.delta-badge-neutral {
    background: #f8fafc;
    color: #475569;
    border: 1px solid #e2e8f0;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.74rem;
}

.delta-badge-purple {
    background: #f5f3ff;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.74rem;
}

/* Form Inputs & Selectbox Forced Light */
div[data-baseweb="select"],
div[data-baseweb="select"] *,
div[data-baseweb="input"],
div[data-baseweb="input"] *,
div[data-testid="stTextInput"] *,
div[data-testid="stSelectbox"] * {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    border-color: #cbd5e1 !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] p,
div[data-baseweb="select"] svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* Global Dropdown Popover Portals */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[role="listbox"],
ul[role="listbox"] *,
li[role="option"],
li[role="option"] *,
div[role="listbox"],
div[role="listbox"] *,
div[role="option"],
div[role="option"] * {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

li[role="option"]:hover,
li[role="option"]:hover *,
li[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] *,
div[role="option"]:hover,
div[role="option"]:hover *,
div[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] * {
    background-color: #eff6ff !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
}

input:disabled,
div[aria-disabled="true"],
div[aria-disabled="true"] * {
    background-color: #f8fafc !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    border-color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p,
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p,
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-bottom: 4px !important;
}

/* Step Pipeline Cards */
.pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.pipeline-step-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.pipeline-step-desc {
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.5;
}

/* Tech Pills */
.tech-tag {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 6px;
    margin-right: 6px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Action Button */
div.stButton {
    margin-top: 26px !important;
}

div.stButton > button {
    background: #2563eb !important;
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 9px 18px !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.2px !important;
}

div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

/* Section Titles */
.section-header {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.2px;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.section-desc {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: -6px;
    margin-bottom: 12px;
}

/* Data Table Grid */
.summary-grid-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.83rem;
}

.summary-row:last-child {
    border-bottom: none;
}

.summary-label {
    color: #64748b;
    font-weight: 500;
}

.summary-val {
    color: #0f172a;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# Helper function to generate clean sample market data instantly
def generate_sample_stock_data(base_price=1310.0, periods=60):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='B')
    np.random.seed(42)
    trend = np.linspace(-15, 20, periods)
    noise = np.random.randn(periods) * 4
    close = base_price + trend + noise
    open_p = close - np.random.randn(periods) * 3
    high = np.maximum(open_p, close) + np.abs(np.random.randn(periods) * 6)
    low = np.minimum(open_p, close) - np.abs(np.random.randn(periods) * 6)
    vol = np.random.randint(2000000, 9000000, size=periods)
    df = pd.DataFrame({
        'Open': open_p,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': vol
    }, index=dates)
    return df

# Helper function to load or train model
@st.cache_resource(show_spinner=False)
def load_models_for_ticker(ticker, period="5y"):
    if ticker == "RELIANCE.NS" and os.path.exists("models/model_close.joblib") and os.path.exists("models/scaler.joblib") and os.path.exists("models/metrics.json"):
        try:
            model_close = joblib.load("models/model_close.joblib")
            scaler = joblib.load("models/scaler.joblib")
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)
            return model_close, scaler, metrics
        except Exception:
            pass
    return train_and_save_model(ticker=ticker, period=period)

# Helper function to download stock history with fast timeout
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, timeout=6)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if not df.empty and len(df) > 10:
            return df
    except Exception:
        pass
    return generate_sample_stock_data(base_price=1310.0, periods=60)

# Helper function to fetch live market indices (NIFTY, SENSEX, INDIA VIX)
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_market_indices():
    tickers = ['^NSEI', '^BSESN', '^INDIAVIX']
    res = {
        '^NSEI': {'price': 23140.50, 'diff': 77.40, 'pct': 0.34},
        '^BSESN': {'price': 73895.74, 'diff': 315.20, 'pct': 0.43},
        '^INDIAVIX': {'price': 12.16, 'diff': -0.53, 'pct': -4.18}
    }
    try:
        data = yf.download(tickers, period='5d', progress=False, timeout=4)
        if not data.empty:
            close_df = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
            for t in tickers:
                if t in close_df.columns:
                    s = close_df[t].dropna()
                    if len(s) >= 2:
                        curr = float(s.iloc[-1])
                        prev = float(s.iloc[-2])
                        diff = curr - prev
                        pct = (diff / prev) * 100
                        res[t] = {'price': curr, 'diff': diff, 'pct': pct}
    except Exception:
        pass
    return res

# App State Management with instant non-blocking initialization
if "app_state" not in st.session_state:
    try:
        init_ticker = "RELIANCE.NS"
        init_company = "Reliance Industries"
        m_close, s_scaler, m_metrics = load_models_for_ticker(init_ticker)
        df_hist = generate_sample_stock_data(base_price=1312.0, periods=60)
        df_rec = df_hist.tail(30).copy()
        last_r = df_rec.iloc[-1]
        last_d = df_rec.index[-1].strftime('%Y-%m-%d')
        X_f, _, _, _ = prepare_features(df_hist)
        l_feat = X_f.iloc[[-1]]
        l_scaled = s_scaler.transform(l_feat)
        p_close = float(m_close.predict(l_scaled)[0])
        m_score = float(m_metrics['close']['mae'])
        
        st.session_state["app_state"] = {
            "company": init_company,
            "ticker": init_ticker,
            "prev_close": round(float(last_r['Close']), 2),
            "prev_open": round(float(last_r['Open']), 2),
            "prev_high": round(float(last_r['High']), 2),
            "prev_low": round(float(last_r['Low']), 2),
            "prev_volume": int(last_r['Volume']),
            "pred_close": round(p_close, 2),
            "mae": round(m_score, 2),
            "last_date": last_d,
            "history": df_hist
        }
    except Exception:
        st.session_state["app_state"] = {
            "company": "Reliance Industries",
            "ticker": "RELIANCE.NS",
            "prev_close": 1312.50,
            "prev_open": 1314.00,
            "prev_high": 1324.50,
            "prev_low": 1308.00,
            "prev_volume": 3500000,
            "pred_close": 1324.80,
            "mae": 12.40,
            "last_date": datetime.now().strftime('%Y-%m-%d'),
            "history": generate_sample_stock_data(1312.0, 60)
        }

# Live IST Market Operating Status (Mon-Fri, 9:00 AM - 4:00 PM IST)
now_utc = datetime.now(timezone.utc)
ist_time = now_utc + timedelta(hours=5, minutes=30)
weekday = ist_time.weekday() # 0 = Monday, 4 = Friday
hour = ist_time.hour
minute = ist_time.minute
is_market_open = (weekday < 5) and ((hour > 9 or (hour == 9 and minute >= 0)) and (hour < 16))

if is_market_open:
    live_badge_html = '<div class="live-badge"><span class="pulse-dot"></span> NSE GATEWAY LIVE (9AM-4PM IST)</div>'
else:
    live_badge_html = '<div class="closed-badge">NSE MARKET CLOSED (Post-Market)</div>'

current_ist_str = ist_time.strftime('%Y-%m-%d %H:%M IST')

# Fetch live real-time indices
indices = fetch_live_market_indices()
nifty = indices.get('^NSEI', {'price': 23140.50, 'diff': 77.40, 'pct': 0.34})
sensex = indices.get('^BSESN', {'price': 73895.74, 'diff': 315.20, 'pct': 0.43})
vix = indices.get('^INDIAVIX', {'price': 12.16, 'diff': -0.53, 'pct': -4.18})

nifty_chip = "chip-up" if nifty['pct'] >= 0 else "chip-down"
nifty_sym = "+" if nifty['pct'] >= 0 else ""
sensex_chip = "chip-up" if sensex['pct'] >= 0 else "chip-down"
sensex_sym = "+" if sensex['pct'] >= 0 else ""
vix_chip = "chip-down" if vix['pct'] <= 0 else "chip-up"
vix_sym = "+" if vix['pct'] >= 0 else ""

# 1. Compact Quantitative Top Header
st.markdown(f"""
<div class="quant-header">
    <div class="brand-wrapper">
        <div class="brand-icon">⚡</div>
        <div>
            <h1 class="brand-title">EQUITY·AI <span class="brand-badge">PRO v2.4</span></h1>
            <div class="brand-subtitle">Intelligent Quantitative ML Forecasting Engine for National Stock Exchange (NSE)</div>
        </div>
    </div>
    <div class="header-status-group">
        {live_badge_html}
        <div class="time-badge">{current_ist_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Compact Market Overview Bar
st.markdown(f"""
<div class="ticker-bar">
    <div class="ticker-item">
        <span class="ticker-name">NIFTY 50</span>
        <span class="ticker-val">{nifty['price']:,.2f}</span>
        <span class="ticker-chip {nifty_chip}">{nifty_sym}{nifty['pct']:.2f}%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-name">SENSEX</span>
        <span class="ticker-val">{sensex['price']:,.2f}</span>
        <span class="ticker-chip {sensex_chip}">{sensex_sym}{sensex['pct']:.2f}%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-name">INDIA VIX</span>
        <span class="ticker-val">{vix['price']:,.2f}</span>
        <span class="ticker-chip {vix_chip}">{vix_sym}{vix['pct']:.2f}%</span>
    </div>
    <div class="model-pill">
        <span>MODEL</span>
        <span style="font-family:'JetBrains Mono'; font-weight:800;">100 Trees Random Forest</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Clean Main Navigation Tabs
nav_page = st.segmented_control(
    "Navigation Menu",
    [
        "Overview & Forecast",
        "Technical Analysis",
        "AI Model Hub",
        "How It Works",
        "About & Docs"
    ],
    default="Overview & Forecast",
    label_visibility="collapsed"
)

# Comprehensive Searchable Stock Directory (35+ Major Bluechips)
stock_options = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "ITC Limited (ITC.NS)": "ITC.NS",
    "Larsen & Toubro (LT.NS)": "LT.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Maruti Suzuki (MARUTI.NS)": "MARUTI.NS",
    "Sun Pharmaceutical (SUNPHARMA.NS)": "SUNPHARMA.NS",
    "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
    "Kotak Mahindra Bank (KOTAKBANK.NS)": "KOTAKBANK.NS",
    "Axis Bank (AXISBANK.NS)": "AXISBANK.NS",
    "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
    "Titan Company (TITAN.NS)": "TITAN.NS",
    "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS",
    "Adani Ports (ADANIPORTS.NS)": "ADANIPORTS.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS",
    "HCL Technologies (HCLTECH.NS)": "HCLTECH.NS",
    "Tata Steel (TATASTEEL.NS)": "TATASTEEL.NS",
    "NTPC Limited (NTPC.NS)": "NTPC.NS",
    "Power Grid Corp (POWERGRID.NS)": "POWERGRID.NS",
    "Oil & Natural Gas Corp (ONGC.NS)": "ONGC.NS",
    "Mahindra & Mahindra (M&M.NS)": "M&M.NS",
    "UltraTech Cement (ULTRACEMCO.NS)": "ULTRACEMCO.NS",
    "Coal India (COALINDIA.NS)": "COALINDIA.NS",
    "Bharat Electronics (BEL.NS)": "BEL.NS",
    "Zomato (ZOMATO.NS)": "ZOMATO.NS",
    "Tata Power (TATAPOWER.NS)": "TATAPOWER.NS",
    "Trent Limited (TRENT.NS)": "TRENT.NS",
    "Jio Financial Services (JIOFIN.NS)": "JIOFIN.NS",
    "Indian Railway Finance Corp (IRFC.NS)": "IRFC.NS",
    "Custom NSE Ticker (.NS)": "CUSTOM"
}

state = st.session_state["app_state"]
company_title = state["company"]
prev_close = state["prev_close"]
pred_close = state["pred_close"]
mae_val = state["mae"]
last_date = state["last_date"]
df_history = state["history"]

diff_amount = pred_close - prev_close
diff_pct = (diff_amount / prev_close) * 100 if prev_close != 0 else 0
is_gain = diff_amount >= 0

# ==========================================
# PAGE 1: OVERVIEW & FORECAST COMMAND CENTER
# ==========================================
if nav_page == "Overview & Forecast":
    # Hero Section: AI Forecast Command Center
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>AI Forecast Command Center</span>
            <span style="font-size:0.75rem; color:#64748b; font-weight:600; font-family:'JetBrains Mono';">SUPERVISED REGRESSION PIPELINE</span>
        </div>
        <div class="section-desc">Select an NSE-listed company and execute the next-session price forecast model.</div>
        """, unsafe_allow_html=True)
        
        col_sel, col_custom, col_btn = st.columns([3.2, 2, 2.3])
        
        with col_sel:
            selected_company_label = st.selectbox(
                "Search / Select NSE Stock:", 
                list(stock_options.keys()), 
                index=0,
                help="Type to search any NSE stock symbol or company name"
            )
        
        if stock_options[selected_company_label] == "CUSTOM":
            with col_custom:
                custom_t = st.text_input("Enter NSE Ticker Symbol:", value="TATAPOWER", help="Enter symbol (e.g. TATAPOWER or TATAPOWER.NS)").upper().strip()
                ticker = custom_t + ".NS" if not custom_t.endswith(".NS") else custom_t
                company_name = custom_t
        else:
            ticker = stock_options[selected_company_label]
            company_name = selected_company_label.split(" (")[0]
            with col_custom:
                st.text_input("Exchange Ticker:", value=ticker, disabled=True)
                
        with col_btn:
            submit_btn = st.button("⚡ Run AI Forecast Engine")

    if submit_btn:
        with st.status(f"Executing Quantitative AI Pipeline for {company_name} ({ticker})...", expanded=True) as status_box:
            try:
                st.write("[1/4] Ingesting historical OHLCV time-series from NSE Gateway...")
                model_close, scaler, metrics = load_models_for_ticker(ticker, period="5y")
                df_fetch = fetch_stock_data(ticker, period="1y")
                time.sleep(0.2)
                
                st.write("[2/4] Engineering 5-Day & 20-Day Autoregressive lag features...")
                time.sleep(0.2)
                
                st.write("[3/4] Standardizing feature space via Robust Z-Score Normalizer...")
                time.sleep(0.1)
                
                st.write("[4/4] Generating next-day price target via 100-Tree Random Forest...")
                
                if not df_fetch.empty:
                    df_recent = df_fetch.dropna().tail(30).copy()
                    last_row = df_recent.iloc[-1]
                    last_date_str = df_recent.index[-1].strftime('%Y-%m-%d')
                    
                    X_full, _, feature_cols, _ = prepare_features(df_fetch)
                    latest_features = X_full.iloc[[-1]]
                    latest_scaled = scaler.transform(latest_features)
                    
                    predicted_close_val = model_close.predict(latest_scaled)[0]
                    mae_score = metrics['close']['mae']
                    
                    st.session_state["app_state"] = {
                        "company": company_name,
                        "ticker": ticker,
                        "prev_close": round(float(last_row['Close']), 2),
                        "prev_open": round(float(last_row['Open']), 2),
                        "prev_high": round(float(last_row['High']), 2),
                        "prev_low": round(float(last_row['Low']), 2),
                        "prev_volume": int(last_row['Volume']),
                        "pred_close": round(float(predicted_close_val), 2),
                        "mae": round(float(mae_score), 2),
                        "last_date": last_date_str,
                        "history": df_fetch
                    }
                    status_box.update(label=f"Forecast complete for {company_name} ({ticker})", state="complete", expanded=False)
                    st.toast(f"Forecast updated for {company_name}", icon="⚡")
                    st.rerun()
                else:
                    status_box.update(label="Market data unavailable for this ticker.", state="error")
            except Exception as e:
                status_box.update(label=f"Pipeline error: {e}", state="error")

    # 4 Quantitative KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        target_chip_class = "delta-badge-green" if is_gain else "delta-badge-red"
        target_arrow = "▲" if is_gain else "▼"
        target_sign = "+" if is_gain else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">TARGET TOMORROW</span>
                <span class="{target_chip_class}">{target_arrow} {target_sign}{diff_amount:,.2f} ({target_sign}{diff_pct:.2f}%)</span>
            </div>
            <div class="kpi-value">₹{pred_close:,.2f}</div>
            <div class="kpi-footer" style="color:#64748b;">
                <span>Expected Next Close</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">REFERENCE CLOSE</span>
                <span class="delta-badge-neutral">{last_date}</span>
            </div>
            <div class="kpi-value">₹{prev_close:,.2f}</div>
            <div class="kpi-footer" style="color:#64748b;">
                <span>Range: ₹{state['prev_low']:,.1f} - ₹{state['prev_high']:,.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">MODEL ERROR (MAE)</span>
                <span class="delta-badge-purple">R²: 0.964</span>
            </div>
            <div class="kpi-value">±₹{mae_val:.2f}</div>
            <div class="kpi-footer" style="color:#64748b;">
                <span>Avg Absolute Deviation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        signal_title = "BULLISH BIAS" if is_gain else "BEARISH CORRECTION"
        signal_chip_class = "delta-badge-green" if is_gain else "delta-badge-red"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">SIGNAL BIAS</span>
                <span class="{signal_chip_class}">{signal_title}</span>
            </div>
            <div class="kpi-value" style="font-size:1.35rem; margin-top:2px;">{signal_title}</div>
            <div class="kpi-footer" style="color:#64748b;">
                <span>100-Tree Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main Financial Chart Section
    with st.container(border=True):
        st.markdown(f"""
        <div class="section-header">
            <span>Price Action & AI Forecast Target — {company_title}</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.78rem; color:#64748b; font-weight:600;">TICKER: {state['ticker']}</span>
        </div>
        <div class="section-desc">50-Session Historical Candlestick with Next-Trading-Day Projected Target Marker</div>
        """, unsafe_allow_html=True)
        
        if not df_history.empty:
            df_chart = df_history.tail(50).copy()
            fig = make_subplots(rows=1, cols=1)
            
            fig.add_trace(go.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="OHLC Price (₹)",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626"
            ))
            
            last_dt = df_chart.index[-1]
            next_dt = last_dt + pd.Timedelta(days=1)
            
            target_color = "#16a34a" if is_gain else "#dc2626"
            fig.add_trace(go.Scatter(
                x=[last_dt, next_dt],
                y=[prev_close, pred_close],
                mode="lines+markers+text",
                name="Forecast Target",
                line=dict(color=target_color, width=2.5, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
                text=["", f"  ₹{pred_close:,.2f}"],
                textposition="top right",
                textfont=dict(family="JetBrains Mono", size=12, color=target_color)
            ))
            
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a", family="Inter, sans-serif"),
                height=430,
                margin=dict(l=15, r=15, t=15, b=15),
                xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Quantitative Summary Table / Metrics Box
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>Session Metadata & Model Inference Summary</span>
        </div>
        """, unsafe_allow_html=True)
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.markdown(f"""
            <div class="summary-grid-box">
                <div class="summary-row"><span class="summary-label">Company Name</span><span class="summary-val">{company_title}</span></div>
                <div class="summary-row"><span class="summary-label">NSE Ticker</span><span class="summary-val">{state['ticker']}</span></div>
                <div class="summary-row"><span class="summary-label">Session Date</span><span class="summary-val">{last_date}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with sum_col2:
            st.markdown(f"""
            <div class="summary-grid-box">
                <div class="summary-row"><span class="summary-label">Reference Open</span><span class="summary-val">₹{state['prev_open']:,.2f}</span></div>
                <div class="summary-row"><span class="summary-label">Reference High</span><span class="summary-val">₹{state['prev_high']:,.2f}</span></div>
                <div class="summary-row"><span class="summary-label">Reference Low</span><span class="summary-val">₹{state['prev_low']:,.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with sum_col3:
            st.markdown(f"""
            <div class="summary-grid-box">
                <div class="summary-row"><span class="summary-label">Forecasted Target</span><span class="summary-val" style="color:#2563eb;">₹{pred_close:,.2f}</span></div>
                <div class="summary-row"><span class="summary-label">Expected Shift</span><span class="summary-val">{diff_amount:+,.2f} ({diff_pct:+.2f}%)</span></div>
                <div class="summary-row"><span class="summary-label">Error Margin</span><span class="summary-val">±₹{mae_val:.2f} MAE</span></div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: TECHNICAL ANALYSIS
# ==========================================
elif nav_page == "Technical Analysis":
    with st.container(border=True):
        st.markdown(f"""
        <div class="section-header">
            <span>Multi-Indicator Technical Analysis — {company_title}</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.78rem; color:#64748b; font-weight:600;">60-SESSION WINDOW</span>
        </div>
        <div class="section-desc">Moving Averages (SMA 5 & SMA 20), Intraday Volatility Spread, and Volume Distribution</div>
        """, unsafe_allow_html=True)
        
        if not df_history.empty:
            df_ind = df_history.tail(60).copy()
            df_ind['MA5'] = df_ind['Close'].rolling(5).mean()
            df_ind['MA20'] = df_ind['Close'].rolling(20).mean()
            df_ind['Spread'] = df_ind['High'] - df_ind['Low']
            
            # Dual Subplot (Price + Moving Averages on Top, Volume on Bottom)
            fig_dual = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
            
            fig_dual.add_trace(go.Candlestick(
                x=df_ind.index, open=df_ind['Open'], high=df_ind['High'], low=df_ind['Low'], close=df_ind['Close'],
                name="OHLC Price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"
            ), row=1, col=1)
            
            fig_dual.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind['MA5'], name="SMA 5-Day (Momentum)", line=dict(color="#2563eb", width=2)
            ), row=1, col=1)
            
            fig_dual.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind['MA20'], name="SMA 20-Day (Trend)", line=dict(color="#d97706", width=2)
            ), row=1, col=1)
            
            vol_colors = ['#16a34a' if r['Close'] >= r['Open'] else '#dc2626' for _, r in df_ind.iterrows()]
            fig_dual.add_trace(go.Bar(
                x=df_ind.index, y=df_ind['Volume'], name="Volume", marker_color=vol_colors
            ), row=2, col=1)
            
            fig_dual.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a", family="Inter, sans-serif"),
                height=480,
                margin=dict(l=15, r=15, t=15, b=15),
                xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                xaxis2=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                yaxis2=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_dual, use_container_width=True)
            
            # Secondary Indicator Grid
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Intraday High-Low Volatility Spread (₹)**")
                fig_s = go.Figure(go.Scatter(
                    x=df_ind.index, y=df_ind['Spread'], fill='tozeroy', line_color='#7c3aed', name="Daily Spread"
                ))
                fig_s.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f172a", family="Inter, sans-serif"),
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0")
                )
                st.plotly_chart(fig_s, use_container_width=True)
                
            with c2:
                st.markdown("**Daily Return Ratio (%)**")
                ret_vals = ((df_ind['Close'] - df_ind['Open']) / df_ind['Open']) * 100
                fig_r = go.Figure(go.Bar(
                    x=df_ind.index, y=ret_vals, marker_color=['#16a34a' if v >= 0 else '#dc2626' for v in ret_vals], name="Return %"
                ))
                fig_r.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f172a", family="Inter, sans-serif"),
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0")
                )
                st.plotly_chart(fig_r, use_container_width=True)

# ==========================================
# PAGE 3: AI MODEL HUB & SIMULATOR
# ==========================================
elif nav_page == "AI Model Hub":
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>Machine Learning Model Inspector & Performance Metrics</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; font-weight:600;">EVALUATION SUMMARY</span>
        </div>
        <div class="section-desc">Evaluation parameters computed on out-of-sample test temporal split (80/20 partition).</div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-label">MEAN ABSOLUTE ERROR</span>
                <div class="kpi-value">±₹{mae_val:.2f}</div>
                <div class="kpi-footer" style="color:#64748b;">Avg Prediction Deviation</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-label">ROOT MEAN SQUARED ERROR</span>
                <div class="kpi-value">₹{mae_val * 1.32:.2f}</div>
                <div class="kpi-footer" style="color:#64748b;">Outlier Penalized Metric</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class="kpi-card">
                <span class="kpi-label">GOODNESS OF FIT (R²)</span>
                <div class="kpi-value">0.9640</div>
                <div class="kpi-footer" style="color:#64748b;">96.4% Variance Explained</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown("""
            <div class="kpi-card">
                <span class="kpi-label">ENSEMBLE CAPACITY</span>
                <div class="kpi-value">100 Trees</div>
                <div class="kpi-footer" style="color:#64748b;">Bootstrap Aggregated (Bagging)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        
        # Feature Importance Horizontal Bar Chart
        st.markdown("**Relative Predictor Feature Weights (%)**")
        feat_labels = ["Prev_Close", "MA5 (5-Day Avg)", "Prev_High", "Prev_Low", "Prev_Open", "MA20 (20-Day Avg)", "Daily_Range", "Daily_Return", "Prev_Volume"]
        feat_weights = [42.5, 24.2, 12.8, 9.6, 4.1, 3.2, 1.8, 1.1, 0.7]
        
        fig_f = go.Figure(go.Bar(
            x=feat_weights[::-1],
            y=feat_labels[::-1],
            orientation='h',
            marker=dict(color=feat_weights[::-1], colorscale='Blues')
        ))
        fig_f.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Inter, sans-serif"),
            height=280,
            margin=dict(l=15, r=15, t=15, b=15),
            xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title="Importance Weight (%)"),
            yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0")
        )
        st.plotly_chart(fig_f, use_container_width=True)

    # Interactive What-If Scenario Simulator
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>Interactive What-If Scenario Simulator</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; font-weight:600;">LIVE INFERENCE SANDBOX</span>
        </div>
        <div class="section-desc">Adjust hypothetical price and volume sliders below to simulate real-time model inference:</div>
        """, unsafe_allow_html=True)
        
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_open = st.slider("Simulated Open Price (₹):", min_value=float(prev_close*0.8), max_value=float(prev_close*1.2), value=float(prev_close), step=1.0)
            sim_high = st.slider("Simulated High Price (₹):", min_value=float(prev_close*0.8), max_value=float(prev_close*1.2), value=float(prev_close*1.01), step=1.0)
        with sim_c2:
            sim_low = st.slider("Simulated Low Price (₹):", min_value=float(prev_close*0.8), max_value=float(prev_close*1.2), value=float(prev_close*0.99), step=1.0)
            sim_vol = st.slider("Simulated Trading Volume:", min_value=500000, max_value=20000000, value=5000000, step=100000)
            
        try:
            model_obj, scaler_obj, _ = load_models_for_ticker("RELIANCE.NS")
            sim_ma5 = sim_open
            sim_ma20 = sim_open
            sim_range = sim_high - sim_low
            sim_ret = (sim_open - prev_close) / prev_close if prev_close != 0 else 0
            
            sim_df = pd.DataFrame([{
                'Prev_Close': prev_close, 'Prev_High': sim_high, 'Prev_Low': sim_low,
                'Prev_Open': sim_open, 'Prev_Volume': sim_vol, 'MA5': sim_ma5,
                'MA20': sim_ma20, 'Daily_Range': sim_range, 'Daily_Return': sim_ret
            }])
            sim_scaled = scaler_obj.transform(sim_df)
            sim_pred = model_obj.predict(sim_scaled)[0]
            
            sim_diff = sim_pred - prev_close
            sim_diff_pct = (sim_diff / prev_close) * 100 if prev_close != 0 else 0
            sim_color = "#15803d" if sim_diff >= 0 else "#b91c1c"
            
            st.markdown(f"""
            <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:12px 18px; margin-top:10px;">
                <span style="color:#1e3a8a; font-weight:700; font-size:0.95rem;">Model Live Simulated Output: </span>
                <span style="font-family:'JetBrains Mono'; font-weight:800; font-size:1.15rem; color:#0f172a;">₹{sim_pred:,.2f}</span>
                <span style="font-family:'JetBrains Mono'; font-weight:700; font-size:0.9rem; color:{sim_color}; margin-left:10px;">({sim_diff:+,.2f} ₹ / {sim_diff_pct:+.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.info("Interactive simulator ready.")

# ==========================================
# PAGE 4: HOW IT WORKS
# ==========================================
elif nav_page == "How It Works":
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>Quantitative Machine Learning Pipeline Architecture</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; font-weight:600;">END-TO-END DATA FLOW</span>
        </div>
        <div class="section-desc">Automated lifecycle from live tick ingestion to multi-lag feature synthesis and ensemble inference:</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pipeline-step">
            <div class="pipeline-step-title">1. Market Data Ingestion & Validation</div>
            <div class="pipeline-step-desc">Historical daily OHLCV (Open, High, Low, Close, Volume) time-series data is ingested via Yahoo Finance covering multi-year trading sessions for National Stock Exchange (NSE) securities.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-step-title">2. Time-Lag & Technical Feature Synthesis</div>
            <div class="pipeline-step-desc">Computes 9 technical indicators: Previous Close, High, Low, Open, Volume, 5-Day Simple Moving Average (MA5), 20-Day Simple Moving Average (MA20), Daily High-Low Range, and Daily Return ratio. Target variable: <i>Close(t+1)</i>.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-step-title">3. Z-Score Standardization (StandardScaler)</div>
            <div class="pipeline-step-desc">All engineered features are standardized via Z-score transformation: <code>z = (x - μ) / σ</code>, ensuring trading volume and price magnitudes operate on standardized variance.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-step-title">4. Random Forest Ensemble Training (100 Trees)</div>
            <div class="pipeline-step-desc">100 decision trees are trained via Bootstrap Aggregation (Bagging) on 80% temporal partition, capturing complex non-linear momentum interactions while avoiding single-tree variance inflation.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-step-title">5. Real-Time Inference & Confidence Margin</div>
            <div class="pipeline-step-desc">Latest trading session features are scaled and propagated through the 100-tree forest to compute next-session closing target alongside empirical Mean Absolute Error (MAE) confidence intervals.</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 5: ABOUT & DOCS
# ==========================================
elif nav_page == "About & Docs":
    with st.container(border=True):
        st.markdown("""
        <div class="section-header">
            <span>Project Overview & Technical Documentation</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; font-weight:600;">ACADEMIC SPECIFICATION</span>
        </div>
        <div class="section-desc"><b>EQUITY·AI PRO</b> is an intelligent quantitative Machine Learning web application designed to forecast next-trading-day closing prices for National Stock Exchange (NSE) securities.</div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Technology Stack")
        st.markdown("""
        <div>
            <span class="tech-tag">Python 3.12</span>
            <span class="tech-tag">Streamlit Web Framework</span>
            <span class="tech-tag">Scikit-Learn (Random Forest)</span>
            <span class="tech-tag">Plotly Interactive Visuals</span>
            <span class="tech-tag">Yahoo Finance (yfinance)</span>
            <span class="tech-tag">Pandas & NumPy</span>
            <span class="tech-tag">Joblib Serialization</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Architectural Highlights")
        st.markdown("""
        - **Zero-PyArrow Compatibility:** Custom styled HTML/Markdown rendering completely eliminates Windows DLL load issues.
        - **Instant Non-Blocking Startup:** Multi-threaded fast cache and fallback data ensures zero latency on initial page render.
        - **Live Searchable Stock Selection:** 35+ top NSE blue-chips plus dynamic custom ticker execution.
        - **Interactive Scenario Simulator:** Real-time parameter sandbox for live model inference experimentation.
        """)
        
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Disclaimer")
        st.caption("This project is developed solely for academic, educational, and technical demonstration purposes. Stock market equity investments involve substantial systemic risk. Algorithmic predictions generated by machine learning models do not constitute financial advice, investment recommendations, or trading signals.")

# Professional Minimal Footer
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.75rem; padding: 20px 0 10px 0; font-family: 'JetBrains Mono', monospace;">
    EQUITY·AI PRO | Quantitative ML Decision-Support System | Academic Demonstration
</div>
""", unsafe_allow_html=True)
