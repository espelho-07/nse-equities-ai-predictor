import os
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    """
    Given a dataframe with columns: Open, High, Low, Close, Volume,
    generates lag features and target for predicting tomorrow's Closing Price.
    """
    data = df.copy()
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
            
    data = data.sort_index()

    # Target: Tomorrow's Close Price
    data['Tomorrow_Close'] = data['Close'].shift(-1)

    # Feature Engineering
    data['Prev_Close'] = data['Close']
    data['Prev_High'] = data['High']
    data['Prev_Low'] = data['Low']
    data['Prev_Open'] = data['Open']
    data['Prev_Volume'] = data['Volume']
    
    # Derived Technical Indicators
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Daily_Range'] = data['High'] - data['Low']
    data['Daily_Return'] = (data['Close'] - data['Open']) / data['Open']

    # Drop NaNs
    data = data.dropna()
    
    feature_cols = [
        'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume',
        'MA5', 'MA20', 'Daily_Range', 'Daily_Return'
    ]
    
    X = data[feature_cols]
    y_close = data['Tomorrow_Close']
    
    return X, y_close, feature_cols, data

def train_and_save_model(ticker="RELIANCE.NS", period="5y"):
    """
    Downloads stock data, trains RF Regressor for Tomorrow Close, and saves artifacts.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False)
    
    if df.empty:
        print(f"Failed to download data for {ticker}. Using fallback synthetic data.")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000)
        np.random.seed(42)
        base_price = 1500 + np.cumsum(np.random.randn(1000) * 10)
        df = pd.DataFrame({
            'Open': base_price + np.random.randn(1000)*5,
            'High': base_price + np.abs(np.random.randn(1000)*15),
            'Low': base_price - np.abs(np.random.randn(1000)*15),
            'Close': base_price + np.random.randn(1000)*5,
            'Volume': np.random.randint(1000000, 10000000, size=1000)
        }, index=dates)

    X, y_close, feature_cols, data = prepare_features(df)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_close.iloc[:split_idx], y_close.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model for Tomorrow Close
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X_train_scaled, y_train)
    pred_close = model_close.predict(X_test_scaled)

    metrics = {
        "close": {
            "mae": round(float(mean_absolute_error(y_test, pred_close)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred_close))), 2),
            "r2": round(float(r2_score(y_test, pred_close)), 4)
        },
        "feature_cols": feature_cols,
        "ticker": ticker,
        "last_updated": str(pd.Timestamp.now())
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_close, "models/model_close.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Model training completed successfully for Tomorrow Closing Price!")
    return model_close, scaler, metrics

if __name__ == "__main__":
    train_and_save_model()
