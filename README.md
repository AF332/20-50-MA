# Advanced Stock Price Prediction Using LSTM and Technical Analysis

## Overview

This project develops a sophisticated tool for forecasting future stock prices of Abbott Laboratories using an array of technical indicators and a deep learning approach. The project harnesses Python's powerful libraries including yfinance for data extraction, TensorFlow and Keras for machine learning, and Plotly alongside Dash for interactive visualisation.

## Data Extraction

Historical stock data for Abbott Laboratories was sourced from Yahoo Finance covering the period from August 2014 to August 2024. This comprehensive dataset includes daily stock prices and volumes which serve as the basis for our analysis and modeling.

## Feature Engineering and Technical Analysis

The implementation of technical indicators such as Moving Averages (MA20, MA50), Exponential Moving Averages (EMA12, EMA26), Moving Average Convergence Divergence (MACD), and Bollinger Bands provided a nuanced view of market trends and volatility. These indicators were pivotal for determining optimal buy and sell points within the stock’s price history, enhancing the strategy’s reactive capability to market conditions.

![image](https://github.com/user-attachments/assets/0f26722e-8e7e-4ec2-9cae-688e62d9477d)

## Predictive Modeling with LSTM

A Long Short-Term Memory (LSTM) model was constructed using a sequential architecture to predict future price movements based on historical data. The model was trained with features including open, high, low, close prices, volume, and technical indicators, ensuring a robust learning process with nuances captured in historical trends.

## Model Evaluation

The model’s performance was rigorously evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). After training for 50 epochs, the model achieved an MAE of approximately 1.872 and an RMSE of approximately 2.016, validating the model's accuracy in predicting stock prices. These metrics helped quantify the accuracy of predictions, ensuring the model's reliability.

![image](https://github.com/user-attachments/assets/a4b045f3-02bb-4be7-b8ff-1883a8232766)


## Visualization and Interactive Application

Dynamic visualisations were created using Plotly, illustrating both the technical analysis indicators and the predictive results. A notable visualisation compares actual stock prices against the predicted values from the model, highlighting the efficacy and precision of the predictions. Additionally, a dashboard was developed using Dash, which provides an interactive platform for real-time data exploration and monitoring.

## Results

The predictive model demonstrated high accuracy, closely mirroring actual stock movements as seen in the plotted results. The actual vs. predicted price graph for Abbott showcases the model’s ability to capture significant price trends and reversals, while the stock analysis graph with technical indicators confirms the model's competency in utilising these metrics for successful price prediction.

![image](https://github.com/user-attachments/assets/74f85851-bdd2-415d-95ba-1220dbad25af)

## Conclusion

This project encapsulates the integration of traditional financial analysis with cutting-edge machine learning techniques. The successful deployment of the LSTM model along with detailed technical analysis and interactive visualizations illustrates a forward-thinking approach to stock market predictions. This tool not only aids investors and analysts in making informed decisions but also sets a benchmark for future developments in the financial modeling domain.
