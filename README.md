# 📈 EQUITY·AI PRO — NSE Equities Machine Learning Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Next-Generation Machine Learning Decision Support & Next-Day Close Price Forecasting System for National Stock Exchange (NSE) Equities.**

---

## 🌟 Key Highlights & Features

- **🎯 Real-Time Multi-Asset Forecasting:** Ingests live OHLCV market feeds from Yahoo Finance for top NSE equities (Reliance, TCS, INFY, HDFC Bank, etc.) or any custom .NS ticker.
- **🌲 Multi-Lag Random Forest Regressor:** 100-tree ensemble model utilizing 9 autoregressive and momentum features (5-Day SMA, 20-Day SMA, Daily Spread, Daily Returns, Historical Lags).
- **📊 Interactive Technical Analysis Suite:** Subplots with Candlestick charts, Moving Average overlays, Volume distribution, and High-Low volatility area charts powered by Plotly.
- **🧪 Interactive "What-If" Scenario Simulator:** Live slider controls to dynamically simulate custom market conditions and observe real-time model inference.
- **⚡ Zero-Latency Instant Rendering:** Multi-threaded non-blocking fallback and caching mechanisms for lightning-fast page loading.
- **🛡️ 100% Zero-PyArrow Architecture:** Custom styled HTML/Markdown table rendering completely eliminating C++ DLL runtime dependencies.

---

## 🏗️ Architecture & Pipeline

`mermaid
flowchart LR
    A[Yahoo Finance Live Market Feed] --> B[Data Ingestion & Cleaning]
    B --> C[Feature Engineering: Lag L1, SMA5, SMA20, Volatility]
    C --> D[Z-Score Standardization: StandardScaler]
    D --> E[Random Forest Regressor: 100 Estimators]
    E --> F[Tomorrow's Price Projection + MAE Confidence Intervals]
    F --> G[Interactive Streamlit Dashboard]
`

---

## 🚀 Local Installation & Setup

1. **Clone the Repository:**
   `ash
   git clone https://github.com/espelho-07/nse-equities-ai-predictor.git
   cd nse-equities-ai-predictor
   `

2. **Create and Activate a Virtual Environment:**
   `ash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   `

3. **Install Dependencies:**
   `ash
   pip install -r requirements.txt
   `

4. **Launch the Application:**
   `ash
   streamlit run app.py
   `

5. Open your browser at http://localhost:8501.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend UI** | Streamlit 1.35+ |
| **Machine Learning** | Scikit-Learn (RandomForestRegressor), Joblib |
| **Market Data** | Yahoo Finance (yfinance) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Graph Objects |

---

## ⚠️ Academic Disclaimer

*This application is developed strictly for educational and academic demonstration purposes. Machine learning stock predictions involve systemic financial risk and should never be considered financial, investment, or trading advice.*
