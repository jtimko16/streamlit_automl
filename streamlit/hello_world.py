import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Function to fetch stock data
def get_stock_data(ticker):
  try:
    data = yf.download(ticker, period="1y")
    return data
  except:
    st.error(f"Error fetching data for {ticker}")
    return None

# Function to train and predict
def predict_stock_price(data):
  # Prepare data for training
  X = data.iloc[:-1, [data.columns.get_loc("Close")]]
  y = data.iloc[1:, [data.columns.get_loc("Close")]]

  # Train model (linear regression)
  model = LinearRegression()
  model.fit(X, y)

  # Predict next day's closing price
  predicted_price = model.predict([[data['Close'].iloc[-1]]])[0]
  return predicted_price

st.title("Stock Price Prediction App")

# Company selection dropdown
ticker = st.selectbox("Select Company", list(yf.Tickers().tickers.keys()))

# Button to trigger data fetch and prediction
if st.button("Get Stock Data & Predict"):
  data = get_stock_data(ticker)
  if data is not None:
    # Display closing price data
    st.write(f"Closing Prices for {ticker}")
    st.line_chart(data["Close"])

    # Make prediction and display result
    predicted_price = predict_stock_price(data.copy())
    st.success(f"Predicted Closing Price for Tomorrow: ${predicted_price:.2f}")

st.info("**Note:** This is a basic example using linear regression. More advanced models and features can be used for better predictions.")
