"""Import all neccessary modules and classes"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash
from dash import html
from dash import dcc

"""Abbott dataset extracted from yahoo finance"""
start = "2014-08-01" # Specified start date of the data extraction.
end = "2024-08-01" # Specified end date of the data extraction.

dataset = yf.download('0Q15.IL', start, end) # Downloads the data using the ticker, start and end date.

"""Adding the moving averages columns"""
dataset['MA20'] = dataset['Adj Close'].rolling(20).mean() # MA20 column is added in and calculated by rolling over adjusted close column with a window of 20 days and the mean is then calculated. This explains why the first 20 data points are provided with NaN.
dataset['MA50'] = dataset['Adj Close'].rolling(50).mean() # Same procedure happens for the MA50 column.

"""Adding the EMA12, EMA26, MACD, and Signal columns into the dataset"""
dataset['EMA12'] = dataset['Adj Close'].ewm(span = 12, adjust = False).mean() # Calculates the 12 period (day) EMA of the adjused close pice. Adjus seet o false for recursive calculation of EMA, which give higher weight to more recent data. The mean is then calculated over the specified span.
dataset['EMA26'] = dataset['Adj Close'].ewm(span = 26, adjust = False).mean() # Same process as  the 12 EMA but this time with a period of 26.
dataset['MACD'] = dataset['EMA12'] - dataset['EMA26'] # Calculates the MACD.
dataset['Signal'] = dataset['MACD'].ewm(span = 9, adjust = False).mean() # Calculates the EMA of the MACD values using a period of 9 days.


"""Adding the Middle, Upper, and Lower bands columns"""
dataset['Middle Band'] = dataset['Adj Close'].rolling(20).mean() # Simply a moving average of the last 20 periods of the adjusted close prices.
dataset['Upper Band'] = dataset['Middle Band'] + 2 * dataset['Adj Close'].rolling(20).std() # Starts with the middle band value and adds twice the standard deviation of the adj close prices over the past 20 periods to the middle band. This std measures price volatility, and multiplying it by 2 scales the measure to capture typical price movements away from the average.
dataset['Lower Band'] = dataset['Middle Band'] - 2 * dataset['Adj Close'].rolling(20).std() # The same as the upper band but the opposite (by minusing) to encompass the lower range of typicl price movements.

"""Creating the buy and sell lists"""
Buy = []
Sell = [] 

"""Creating the logic of when to buy and sell"""
for i in range (len(dataset)): # Loops over the length of the dataset and 'i' represents the current index in the dataset for each iteration.
    if dataset.MA20.iloc[i] > dataset.MA50.iloc[i] \
    and dataset.MA20.iloc[i-1] < dataset.MA50.iloc[i-1] \
    and dataset['MACD'].iloc[i] > dataset['Signal'].iloc[i]: # Checks if the MA20 is greater than MA50 at the current index 'i' and checks the values at the previous index to determine if MA20 was less than MA50. Also checks if the MACD is higher than the signal line.
        Buy.append(i) # if true append the current index to the buy list.
    elif dataset.MA20.iloc[i] < dataset.MA50.iloc[i] \
    and dataset.MA20.iloc[i-1] > dataset.MA50.iloc[i-1] \
    and dataset['MACD'].iloc[i] < dataset['Signal'].iloc[i]: # Does the same as the previous logic but in reverse for the sell signals.
        Sell.append(i) # if true append the current index to the sell list.

fig = go.Figure(data = [go.Candlestick(x = dataset.index,
                                       open = dataset['Open'],
                                       high = dataset['High'],
                                       low = dataset['Low'],
                                       close = dataset['Adj Close'],
                                       name = 'Candlestick')]) 

"""Plotting MA20, and MA50 lines"""
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['MA20'],
                         mode = 'lines',
                         name = 'MA20',
                         line = dict(color = 'white', width = 2, dash = 'solid'),
                         opacity = 0.5))
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['MA50'],
                         mode = 'lines',
                         name = 'MA50',
                         line = dict(color = 'magenta', width = 2, dash = 'solid'),
                         opacity = 0.5))

"""Plotting the MACD and Signal lines"""
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['MACD'],
                         name = 'MACD',
                         line = dict(color = 'red')))
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['Signal'],
                         name = 'Signal Line',
                         line = dict(color = 'green')))

"""Plotting the Bollinger bands lines"""
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['Upper Band'],
                         name = 'Upper Bollinger Band',
                         line = dict(color = 'yellow')))
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['Middle Band'],
                         name = 'Middle Bollinger Band',
                         line = dict(color = 'blue')))
fig.add_trace(go.Scatter(x = dataset.index,
                         y = dataset['Lower Band'],
                         name = 'Lower Bollinger Band',
                         line = dict(color = 'yellow')))

"""Plotting the locations of the buy and sell markers"""
fig.add_trace(go.Scatter(x = dataset.iloc[Buy].index, 
                         y = dataset.iloc[Buy]['Adj Close'], 
                         mode = 'markers',
                         name = 'Buy Signal', 
                         marker = dict(color = 'green', size = 10, symbol = 'triangle-up')))
fig.add_trace(go.Scatter(x = dataset.iloc[Sell].index,
                         y = dataset.iloc[Sell]['Adj Close'],
                         mode = 'markers',
                         name = 'Sell Signal',
                         marker = dict(color = 'red', size = 10, symbol = 'triangle-down')))

"""Configuring the graph layout"""
fig.update_layout(title = 'Stock Price Analysis for Abbott',
                  xaxis_title = 'Date',
                  yaxis_title = 'Price in Pounds',
                  legend_title = 'Legend',
                  template = 'plotly_dark')

fig.update_xaxes(
    tickangle = 30,
    tickmode = 'auto',
    type = 'date'
)

"""Show the figure"""
fig.show()

"""Creating and Performing LSTM model to predict future price movements"""
dataset['Prev Close'] = dataset['Adj Close'].shift(1) # Prev Close column created. This column is filled with the values from the adj close column, but each value is shifted down by one row.
dataset.dropna(inplace = True) # Removes all NaN values and makes the changes directly to the original dataframe.
print(dataset)

"""Adj Close is the target and the other columns are the features used to predict"""
features = dataset[['Open', 'High', 'Low', 'Close', 'MA20', 'Volume', 'MA50', 'EMA12', 'EMA26', 'Signal', 'Middle Band', 'MACD', 'Prev Close', 'Upper Band', 'Lower Band']] # Selecting all columns beig used as features for the model to learn from.
target = dataset['Adj Close'] # The target of the model.

"""Scale features and target"""
scalar = MinMaxScaler(feature_range = (0, 1)) # Use the MinMaxScalar class to preprocess each feature into range between 0 and 1.
scaled_features = scalar.fit_transform(features) # Fit method calculates the max and min values of 'features' and then the 'transform' uses the value to scales the features to the specified range.
scaled_target = scalar.fit_transform(target.values.reshape(-1, 1)) # 'target.values' extracts the values from the 'target' series as a numpy array. 'reshape(-1,1)' changes the shape of the array to make it a 2D array with one column. '-1' tells numpy to calculate the number of rows based on the length of the array.

"""Prepare data for LSTM"""
train_size = int(len(dataset) * 0.8) # Calculates the number of data points to include in the training set by taking 80% of the total number of rows in the dataset. Then converts it into an integer.
x_train, x_test = scaled_features[:train_size], scaled_features[train_size:] # slices the 'sclaed_features' array to take the first 80% of rows for training and the remaining 20% for testing.
y_train, y_test = scaled_target[:train_size], scaled_target[train_size:] # Does the same as the previous line but for the target data.

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])) # reshapes the x_train into 3D format suitable for input LSTM network. First dimension is the number of samples. Second, represents the number of time steps per sample. Thrd is the number of features in each sample.
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])) # Same process as the x_train.

"""Building the LSTM model"""
model = Sequential([
    LSTM(50, return_sequences = True, input_shape = (1, x_train.shape[2])),
    LSTM(50),
    Dense(1)
]) # Sequential model is initialised, then adds an LSTM layer to the model with 50 units (neurons). 'return_sequence' set to True to ensure that the LSTM layer outputs a sequence rather than a single value at each time step. The input_shape is then defined. where the number of time steps (1) were provided and the number of features per time step (x_tain.shape[2]).

"""Training the model"""
model.compile(loss = 'mean_squared_error', optimizer = 'adam') # configures the model for training by setting the loss function and the optimiser.
model.fit(x_train, y_train, epochs = 50, batch_size = 1, verbose = 2) # Trains the model for a fixed number of epochs. batch_size is the number of samples per gradient update for training. verbose controls the amount of information displayed during training, '2' means less information will be displayed.

"""Testing model for prediction"""
predicted_prices = model.predict(x_test) # predictions generated from the input features using the trained model.
predicted_prices = scalar.inverse_transform(predicted_prices) # Since the predictions are in a scaled format. This converts the predicted values back to its original scale.

"""Computing the metrics"""
actual_price = scalar.inverse_transform(y_test.reshape(-1, 1)) # reshapes the y_test array into 2D with one column. -1 indicates that numpy should automatically determine the number of rows based on the length of y_test.
mae = mean_absolute_error(actual_price, predicted_prices) # calculates the mean absolute error between the actual price and predicted price.
rmse = np.sqrt(mean_squared_error(actual_price, predicted_prices)) # Calculates the rmse from the actual price and predicted price.

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""Extract dates for the test dataset"""
test_dates = dataset.index[train_size:]

"""Plotting results"""
plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_price, color='blue', label='Actual Abbott Stock Price')
plt.plot(test_dates, predicted_prices, color='red', linestyle='--', label='Predicted Abbott Stock Price')
plt.title('Abbott Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

"""Initialize the Dash app"""
app = dash.Dash(__name__)

"""Define the layout of the app"""
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

"""Run the app"""
if __name__ == '__main__':
    app.run_server(debug=True)
