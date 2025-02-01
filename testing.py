import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # Technical Analysis library
from statsmodels.tsa.arima.model import ARIMA


# Function to fetch stock data
@st.cache_data(ttl=300)
def get_stock_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)
    if "Close" not in data:
        return None
    return data["Close"]

# Function to calculate MACD
def calculate_macd(data):
    macd_indicator = ta.trend.MACD(close=data)
    return macd_indicator.macd(), macd_indicator.macd_signal()

# Function to calculate RSI
def calculate_rsi(data):
    rsi_indicator = ta.momentum.RSIIndicator(close=data)
    return rsi_indicator.rsi()

# Function to calculate ARIMA forecast (Simplistic Example)
def calculate_arima_forecast(data, periods=5):
    try:
      model = ARIMA(data, order=(5,1,0)) # Define order of the model. This can be tuned to improve results
      model_fit = model.fit()
      forecast = model_fit.forecast(steps=periods)
      return pd.Series(forecast, index=data.index[-periods:])
    except Exception as e:
      print(f"ARIMA error: {e}")
      return pd.Series([None] * periods, index=data.index[-periods:])


# Simple "probability" function (based on multiple signals)
def calculate_rise_probability(close_prices, macd, macd_signal, rsi, arima_forecast):
    probability = 0
    
    # MACD crossover condition
    if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
      probability += 0.3  # Medium positive signal
    elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
      probability -= 0.2   # Negative signal

    # RSI conditions
    if rsi.iloc[-1] < 30:
      probability += 0.2 # Oversold
    elif rsi.iloc[-1] > 70:
       probability -= 0.1  # Overbought

    # ARIMA forecast condition
    if arima_forecast is not None and arima_forecast.iloc[-1] is not None and arima_forecast.iloc[-1] > close_prices.iloc[-1]:
      probability += 0.4   # Positive forecast
    elif arima_forecast is not None and arima_forecast.iloc[-1] is not None and arima_forecast.iloc[-1] < close_prices.iloc[-1]:
      probability -= 0.1 # Negative forecast
    
    # Normalize and categorize probability
    if probability >= 0.7:
       return "Very High"
    elif probability >= 0.3:
        return "High"
    elif probability > -0.2:
      return "Medium"
    else:
       return "Low"


# Streamlit App
st.title("Potential Rising Stock Dashboard")

# Ticker Selection
st.sidebar.header("Stock Selection")
tickers = st.sidebar.text_area("Enter tickers (comma separated):", value="AAPL,MSFT,GOOG").split(",")
tickers = [ticker.strip().upper() for ticker in tickers]

if tickers:
    try:
        # Fetch Data
        with st.spinner("Downloading Data..."):
            close_data = get_stock_data(tickers)
            if close_data is None:
                 st.error("No Data found for given tickers. Please check the ticker symbol and try again.")
                 st.stop()
        
        # Error handling to make sure close_data is not empty
        if close_data.empty:
            st.error("No Data found for given tickers. Please check the ticker symbol and try again.")
            st.stop()

        # Calculate Indicators
        with st.spinner("Calculating Indicators..."):
           macd_results = close_data.apply(calculate_macd)
           macd_data = macd_results.apply(lambda x: x[0])
           macd_signal_data = macd_results.apply(lambda x: x[1])
           rsi_data = close_data.apply(calculate_rsi)
           arima_forecasts = close_data.apply(calculate_arima_forecast)

        # Calculate Rise Probabilities
        probabilities = {}
        for ticker in tickers:
          probabilities[ticker] = calculate_rise_probability(
              close_data[ticker], 
              macd_data[ticker], 
              macd_signal_data[ticker], 
              rsi_data[ticker],
              arima_forecasts[ticker]
              )

        # Create Table with Tickers and Probabilities
        data = []
        for ticker in tickers:
            data.append({"Ticker": ticker, "Probability": probabilities[ticker]})
        df_proba = pd.DataFrame(data)
        st.dataframe(df_proba, hide_index = True)

        # Stock Selection
        selected_ticker = st.selectbox("Select stock to display chart:", tickers)

        # Create Chart (Historical Price and Indicators)
        if selected_ticker:
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Price", "MACD", "RSI"), vertical_spacing=0.05)

            # Price Chart
            fig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker], name="Close Price", mode="lines"), row=1, col=1)

            # MACD Chart
            fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data[selected_ticker], name="MACD", mode="lines"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macd_signal_data.index, y=macd_signal_data[selected_ticker], name="MACD Signal", mode="lines"), row=2, col=1)

            # RSI Chart
            fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data[selected_ticker], name="RSI", mode="lines"), row=3, col=1)

            fig.update_layout(title_text=f"{selected_ticker} Indicators", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Enter ticker symbols in the sidebar.")