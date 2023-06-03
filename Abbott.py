# All the libraries used within the script are called at the start of the code.
import matplotlib.pyplot as plt # For plotting
import numpy as np # For arithmatic calculation within the code
import pandas as pd # For data frames
import io
import requests
import matplotlib.dates as mdates
import yfinance as yf

# Abbott data set taken downloaded from yahoo finance is read through python
start = '2019-08-01'
end = '2022-08-01'
dataset = yf.download('0Q15.IL', start, end)
print(dataset)

# New columns are added into the dataset
dataset['MA20'] = dataset['Adj Close'].rolling(20).mean() # MA20 column is added in and calculated by rolling over the adjusted close column with a window of 20 days and the mean is then calculated. Hence explains why the first 20 data points will not porvide any numbers.
dataset['MA50'] = dataset['Adj Close'].rolling(50).mean() # MA50 is added in and is calculated the same way as the MA20 apart from this time a with a window of 50 days therefore the first 50 data points will not have any numbers representing it.
print(dataset) # New data frame is displayed

# Buy and Sell values are determined
dataset_2019 = dataset.loc['2019-08-01':'2022-08-01'] # Data frame is focused on the covid period from 1st of august 2019 to 1st of august 2022.
Buy = [] # Empty buy list is created to contain calculated buy values.
Sell = [] # Empty sell list is created to contain calculated sell values.

for i in range (len(dataset_2019)): # i is in the range of the length of the data frame 'dataset_2019'.
    if dataset_2019.MA20.iloc[i] > dataset_2019.MA50.iloc[i] \
    and dataset_2019.MA20.iloc[i-1] < dataset_2019.MA50.iloc[i-1]: # For there to be a buy value the MA20 is above the MA50 but was not the day before which is found by using the previous data points.
        Buy.append(i) # The values that agree with this condition is then added into the buy list.
    elif dataset_2019.MA20.iloc[i] < dataset_2019.MA50.iloc[i] \
    and dataset_2019.MA20.iloc[i-1] > dataset_2019.MA50.iloc[i-1]: # For there to be a sell signal the MA20 is below the MA50 but was not the day before by also using the previous data points.
        Sell.append(i) # The values that agree with this condition is then added into the sell list
    
print(Buy) # Buy list is printed
print(Sell) # Sell list is printed

# Data frame is then plotted into a graph
plt.figure(figsize = (12,5))
plt.plot(dataset_2019['Adj Close'], label = 'Stock Price', color = 'blue', alpha = 0.5)
plt.plot(dataset_2019['MA20'], label = 'MA20', color = 'k', alpha = 0.5)
plt.plot(dataset_2019['MA50'], label = 'MA50', color = 'magenta', alpha = 0.5)
plt.scatter(dataset_2019.iloc[Buy].index, dataset_2019.iloc[Buy]['Adj Close'], marker = '^', color = 'g', s = 100)
plt.scatter(dataset_2019.iloc[Sell].index, dataset_2019.iloc[Sell]['Adj Close'], marker = 'v', color = 'r', s = 100)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price in Pounds')
plt.title('Abbott')

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 120))
plt.gca().xaxis.set_tick_params(rotation = 30)
plt.show()

dataset2 = pd.read_csv(r'E:\3rd Year\Semester 1\Research and Development Skills\Project\J&J stock data.csv', index_col='Date', parse_dates=True)
print(dataset2)

dataset2['MA20'] = dataset2['Adj Close'].rolling(20).mean()
dataset2['MA50'] = dataset2['Adj Close'].rolling(50).mean()
print(dataset2)

dataset2_2019 = dataset2.loc['2019-08-01':'2022-08-01']
Buy2 = []
Sell2 = []

for j in range (len(dataset2_2019)):
    if dataset2_2019.MA20.iloc[j] > dataset2_2019.MA50.iloc[j] \
    and dataset2_2019.MA20.iloc[j-1] < dataset2_2019.MA50.iloc[j-1]:
        Buy2.append(j)
    elif dataset2_2019.MA20.iloc[j] < dataset2_2019.MA50.iloc[j] \
    and dataset2_2019.MA20.iloc[j-1] > dataset2_2019.MA50.iloc[j-1]:
        Sell2.append(j)

print(Buy2)
print(Sell2)

plt.figure(figsize = (12,5))
plt.plot(dataset2_2019['Adj Close'], label = 'Stock Price', color = 'blue', alpha = 0.5)
plt.plot(dataset2_2019['MA20'], label = 'MA20', color = 'k', alpha = 0.5)
plt.plot(dataset2_2019['MA50'], label = 'MA50', color = 'magenta', alpha = 0.5)
plt.scatter(dataset2_2019.iloc[Buy2].index, dataset2_2019.iloc[Buy2]['Adj Close'], marker = '^', color = 'g', s = 100)
plt.scatter(dataset2_2019.iloc[Sell2].index, dataset2_2019.iloc[Sell2]['Adj Close'], marker = 'v', color = 'r', s = 100)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price in Pounds')
plt.title('J&J')

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 120))
plt.gca().xaxis.set_tick_params(rotation = 30)
plt.show()