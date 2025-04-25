import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📊 Stock Price Predictor with Backtesting")

# Sidebar options
st.sidebar.header("Configure Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
time_unit = st.sidebar.selectbox("Choose data range", ["Days", "Months", "Years"])
time_value = st.sidebar.slider(f"Select number of {time_unit.lower()}", 1, 365 if time_unit=="Days" else 24 if time_unit=="Months" else 10, 30)
predict_ahead = st.sidebar.slider("Days to Predict Ahead", 1, 30, 7)

# Convert to valid period string for yfinance
def get_period(unit, value):
    return f"{value}{unit[0].lower()}"

period = get_period(time_unit, time_value)

@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period)
    data = data[['Close']].dropna()
    data['Date'] = data.index
    return data

data = load_data(ticker, period)
st.subheader(f"Historical Closing Prices for {ticker}")
st.line_chart(data[['Close']])

# Feature creation
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
rmse = mean_squared_error(y_test, preds, squared=False)

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
