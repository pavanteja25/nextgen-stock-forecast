import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Feature columns ----------
FEATURE_COLS = [
    "Return_1d",
    "MA_5",
    "MA_10",
    "Vol_5",
    "Vol_10",
    "Close_1d_ago",
    "Close_2d_ago",
    "Close_3d_ago",
]

# ---------- Helper functions ----------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Data must have a 'Date' column.")
    if "Close" not in df.columns:
        raise ValueError("Data must have a 'Close' column.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])
    df = df.drop_duplicates(subset=["Date"])
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["Vol_5"] = df["Return_1d"].rolling(window=5).std()
    df["Vol_10"] = df["Return_1d"].rolling(window=10).std()
    df["Close_1d_ago"] = df["Close"].shift(1)
    df["Close_2d_ago"] = df["Close"].shift(2)
    df["Close_3d_ago"] = df["Close"].shift(3)
    df = df.dropna().reset_index(drop=True)
    return df

def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2):
    X = df[FEATURE_COLS].values
    y = df["Close"].values
    dates = df["Date"]
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates.iloc[split_idx:]
    return X_train, X_test, y_train, y_test, dates_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def train_models(X_train_s, y_train):
    lin = LinearRegression()
    lin.fit(X_train_s, y_train)

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    return lin, rf

def evaluate_model(model, X_test_s, y_test, name: str):
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Preds": preds,
    }

def plot_predictions(dates_test, y_test, preds_lin, preds_rf):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates_test, y_test, label="Actual Close")
    ax.plot(dates_test, preds_lin, label="Linear Regression")
    ax.plot(dates_test, preds_rf, label="Random Forest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs Predicted Closing Prices")
    ax.legend()
    fig.tight_layout()
    return fig

def forecast_next_days(df_features: pd.DataFrame, scaler, model, days_ahead: int = 7):
    df = df_features.copy()
    last_row = df.iloc[-1].copy()
    last_date = df["Date"].iloc[-1]

    forecasts = []

    for i in range(days_ahead):
        feat_row = last_row[FEATURE_COLS].values.reshape(1, -1)
        feat_row_s = scaler.transform(feat_row)
        next_close = model.predict(feat_row_s)[0]

        next_date = last_date + pd.Timedelta(days=1)
        forecasts.append({"Date": next_date, "Predicted_Close": next_close})

        last_row["Close_3d_ago"] = last_row["Close_2d_ago"]
        last_row["Close_2d_ago"] = last_row["Close_1d_ago"]
        last_row["Close_1d_ago"] = next_close

        last_close = df["Close"].iloc[-1]
        last_row["Return_1d"] = (next_close - last_close) / last_close

        last_row["MA_5"] = (df["Close"].tail(4).sum() + next_close) / 5
        last_row["MA_10"] = (df["Close"].tail(9).sum() + next_close) / 10

        last_row["Vol_5"] = df["Close"].pct_change().tail(5).std()
        last_row["Vol_10"] = df["Close"].pct_change().tail(10).std()

        df = pd.concat(
            [df, pd.DataFrame([{"Date": next_date, "Close": next_close}])],
            ignore_index=True,
        )
        last_date = next_date

    return pd.DataFrame(forecasts)

# ---------- Streamlit App ----------
st.set_page_config(page_title="Next-Gen Stock Market Forecasting", layout="wide")
st.title("📈 Next-Gen Stock Market Forecasting using ML Frameworks")

st.write(
    """
This app:
- Fetches stock data automatically from the web (Yahoo Finance via yfinance)
- Engineers simple technical features
- Trains **Linear Regression** and **Random Forest** models
- Evaluates them and forecasts future prices
"""
)

# --- Sidebar: Data Source (Web or CSV) ---
st.sidebar.header("Data Source")
data_option = st.sidebar.radio(
    "Choose Data Input Method:",
    ("Load from Web (yfinance)", "Upload CSV")
)

df_raw = None

if data_option == "Load from Web (yfinance)":
    ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.sidebar.date_input(
        "Start Date", value=pd.to_datetime("2018-01-01")
    )
    end_date = st.sidebar.date_input(
        "End Date", value=pd.to_datetime("today")
    )

    if st.sidebar.button("Fetch Data"):
        df_raw = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [c[0] for c in df_raw.columns]
        df_raw.reset_index(inplace=True)
        st.success(f"Loaded {len(df_raw)} rows for {ticker}")
    else:
        st.info("Click 'Fetch Data' to load stock prices.")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload stock CSV file", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully.")
    else:
        st.warning("Upload a CSV file or switch to web mode.")
        st.stop()

# --- Sidebar: Model settings ---
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
forecast_days = st.sidebar.slider("Days to forecast", 1, 14, 7)
forecast_model_name = st.sidebar.selectbox(
    "Model for forecasting", ["Random Forest", "Linear Regression"]
)

# --- Main processing pipeline ---
try:
    df_clean = clean_data(df_raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

st.subheader("Raw Data (first 5 rows)")
st.dataframe(df_clean.head())

df_features = engineer_features(df_clean)

if len(df_features) < 50:
    st.error("Not enough data after feature engineering. Use more historical data.")
    st.stop()

X_train, X_test, y_train, y_test, dates_test = train_test_split_time(
    df_features, test_size=test_size
)
X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
lin_model, rf_model = train_models(X_train_s, y_train)

res_lin = evaluate_model(lin_model, X_test_s, y_test, "Linear Regression")
res_rf = evaluate_model(rf_model, X_test_s, y_test, "Random Forest")

metrics_df = pd.DataFrame(
    [
        {
            "Model": res_lin["Model"],
            "MAE": round(res_lin["MAE"], 4),
            "RMSE": round(res_lin["RMSE"], 4),
            "R2": round(res_lin["R2"], 4),
        },
        {
            "Model": res_rf["Model"],
            "MAE": round(res_rf["MAE"], 4),
            "RMSE": round(res_rf["RMSE"], 4),
            "R2": round(res_rf["R2"], 4),
        },
    ]
)

st.subheader("Model Performance")
st.dataframe(metrics_df)

fig = plot_predictions(dates_test, y_test, res_lin["Preds"], res_rf["Preds"])
st.subheader("Actual vs Predicted Prices")
st.pyplot(fig)

model_for_forecast = rf_model if forecast_model_name == "Random Forest" else lin_model

st.subheader(f"Next {forecast_days} Days Forecast ({forecast_model_name})")
forecast_df = forecast_next_days(df_features, scaler, model_for_forecast, forecast_days)
st.dataframe(forecast_df)

