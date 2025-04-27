import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📊 Stock Price Predictor with Backtesting")

# Sidebar options
st.sidebar.header("Configure Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
n_days = st.sidebar.slider("Select number of past days", 30, 1825, 365)
predict_ahead = st.sidebar.slider("Days to Predict Ahead", 1, 30, 7)

# Calculate dates
today = date.today()
start_date = today - timedelta(days=n_days)

@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Data loaded from Yahoo Finance")
    if data.empty:
        st.error(f"No data found for ticker '{ticker}' between {start_date} and {end_date}.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        # Flatten multiindex
        data.columns = data.columns.get_level_values(0)
    
    return data


data = load_data(ticker, start_date, today)

if data.empty:
    st.stop()

st.subheader(f"Historical Closing Prices for {ticker}")
st.line_chart(data[['Close']])

# Feature creation
data = data[['Close']].dropna()
data['Date'] = data.index
data['Target'] = data['Close'].shift(-predict_ahead)
data = data.dropna()
data['Day'] = np.arange(len(data))
X = data[['Day']]
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

st.subheader("🔍 Model Backtesting")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Date'].iloc[y_test.index], y_test, label='Actual Prices', color='black')
ax.plot(data['Date'].iloc[y_test.index], preds, label='Predicted Prices', linestyle='--', color='orange')
ax.set_title(f"Backtesting Results (RMSE = {rmse:.2f})")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Predict future
st.subheader("🔮 Predict Future Prices")
future_days = np.array(range(len(data), len(data) + predict_ahead)).reshape(-1, 1)
future_preds = model.predict(future_days)

future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=predict_ahead)
future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data['Date'], data['Close'], label='Historical Prices')
ax2.plot(future_df['Date'], future_df['Predicted_Close'], label='Predicted Future Prices', linestyle='--')
ax2.set_title(f"Future {predict_ahead}-Day Price Prediction")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Prediction Complete. Adjust inputs from sidebar to explore different scenarios.")
