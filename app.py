# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf                # For fetching stock market data
from keras.models import load_model  # To load the trained LSTM model
import streamlit as st               # For building the interactive web app
import matplotlib.pyplot as plt      # For plotting charts

# Load the previously trained LSTM model
model = load_model('Stock Predictions Model.keras')

# Streamlit App Header
st.header('Stock Market Predictor')

# User input for stock symbol
# Default value is 'GOOG'
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Set the date range for historical data
start = '2012-01-01'
end = '2024-12-31'

# Fetch historical stock data using yfinance
data = yf.download(stock, start, end)

# Display the fetched stock data in Streamlit
st.subheader('Stock Data')
st.write(data)

# Split data into training (first 80%) and testing (last 20%) sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Initialize MinMaxScaler to scale data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Take last 100 days of training data and combine with test data
# This ensures we have previous 100 days for first prediction
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# Scale the combined test data
# ⚠ Note: Ideally, we should use scaler.transform(data_test) to avoid data leakage
data_test_scale = scaler.fit_transform(data_test)

# ------------------------- Visualization -------------------------

# Price vs 50-day Moving Average
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='50-Day MA')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig1)

# Price vs 50-day MA vs 100-day MA
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='50-Day MA')
plt.plot(ma_100_days, 'b', label='100-Day MA')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Price vs 100-day MA vs 200-day MA
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='100-Day MA')
plt.plot(ma_200_days, 'b', label='200-Day MA')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# ------------------------- Prepare Test Sequences -------------------------

x = []
y = []

# Create sequences of 100 days for prediction
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

# Convert lists to NumPy arrays for model input
x, y = np.array(x), np.array(y)

# ------------------------- Make Predictions -------------------------

predict = model.predict(x)

# Reverse the scaling to get actual stock prices
scale = 1 / scaler.scale_  # scale factor from MinMaxScaler
predict = predict * scale
y = y * scale

# ------------------------- Plot Predictions -------------------------

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
