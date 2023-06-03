# All the libraries used within the script are called at the start of the code.
import matplotlib.pyplot as plt # For plotting.
import numpy as np # For arithmatic calculation within the code.
import pandas as pd # For data frames.
import matplotlib.dates as mdates # For the framatting of the dates on the graph.
import yfinance as yf # For extracting stock data from yahoo finance.

# Abbott data set taken downloaded from yahoo finance is read through python
start = '2019-08-01' # Specifies the start date for the data download.
end = '2022-08-01' # Specifies the end date for the data download.
dataset = yf.download('0Q15.IL', start, end) # Downloads the Abbott data using the ticker and the specified start and end dates.
print(dataset) # Prints the dataset to check functionality.

# New columns are added into the dataset
dataset['MA20'] = dataset['Adj Close'].rolling(20).mean() # MA20 column is added in and calculated by rolling over the adjusted close column with a window of 20 days and the mean is then calculated. Hence explains why the first 20 data points will not porvide any numbers.
dataset['MA50'] = dataset['Adj Close'].rolling(50).mean() # MA50 is added in and is calculated the same way as the MA20 apart from this time a with a window of 50 days therefore the first 50 data points will not have any numbers representing it.
print(dataset) # New data frame is displayed.

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
        Sell.append(i) # The values that agree with this condition is then added into the sell list.
    
print(Buy) # Buy list is printed.
print(Sell) # Sell list is printed.

# Data frame is then plotted into a graph
plt.figure(figsize = (12,5)) # Specifies figure to be 12 inches wide and 5 inches high.
plt.plot(dataset_2019['Adj Close'], label = 'Stock Price', color = 'blue', alpha = 0.5) # Plots the adjused close data from the dataset_2019. Label is specified to be 'Stock Price' and colour of the graph will be blue. The alpha specifies the transparency of the line of the graph.
plt.plot(dataset_2019['MA20'], label = 'MA20', color = 'k', alpha = 0.5) # The MA20 data for the data_2019 is plotted on the same figure. Label is MA20 and the colour of the line is black. transparency is is get as 0.5.
plt.plot(dataset_2019['MA50'], label = 'MA50', color = 'magenta', alpha = 0.5) # MA50 is plotted with label MA50. Line colour is set to magneta and the transparency is 0.5.
plt.scatter(dataset_2019.iloc[Buy].index, dataset_2019.iloc[Buy]['Adj Close'], marker = '^', color = 'g', s = 100) # Plotting as a scatter plot with x-axis as the index label of the buy values present in the 'dataset_2019' and the y-axis as the adjusted close values for the corresponding buy values calculated. A marker is specifed to represent the buy signal and was choosen to be the colour green. The marker size was also set to 100 by the value of 's'.
plt.scatter(dataset_2019.iloc[Sell].index, dataset_2019.iloc[Sell]['Adj Close'], marker = 'v', color = 'r', s = 100) # Plotting as a scatter plot with x-axis as the index label of the sell values present in the 'dataset_2019' and the y-axis as the adjusted close values for the corresponding sell values calculated. A marker is specifed to represent the buy signal and was choosen to be the colour red. The marker size was also set to 100 by the value of 's'.
plt.legend() # Creates a legend for the graph.
plt.xlabel('Date') # Creates the x-label as 'Dates'.
plt.ylabel('Price in Pounds') # Creates the y-label as 'Price in Pounds'
plt.title('Abbott') # Creates a title as 'Abbott'.

# Formatting the x-axis for a cleaner look
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 120)) # Scales the x-axis with an interval of 120 using the library mdates.
plt.gca().xaxis.set_tick_params(rotation = 30) # Rotates the dates on the x-axis by 30 degrees for aesthetics.
plt.show() # Shows the finished plot.

# J&J data set taken downloaded from yahoo finance is read through python
start2 = '2019-08-01' # Specifies the start date for the data download.
end2 = '2022-08-01' # Specifies the end date for the data download.
dataset2 = yf.download('0R34.IL', start2, end2) # Downloads the J&J data using the ticker and the specified start and end dates.
print(dataset2) # Prints the dataset to check functionality.

# New columns are added into the dataset
dataset2['MA20'] = dataset2['Adj Close'].rolling(20).mean() # 
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