# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:27:32 2023

@author: yunhui
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from dataset import get_stock_data
from modeling import lstm_model, rnn_model, cnn_model



def plot_lstm(df, y_test, lstm_predictions, ticker):
    min_value = df['Close'].min()
    max_value = df['Close'].max()

    feature_index = 0
    lstm_predictions_original = lstm_predictions[:, feature_index] * (max_value - min_value) + min_value
    y_test_original = y_test[:, feature_index] * (max_value - min_value) + min_value
    df['Date'] = pd.to_datetime(df['Date'])

    df_subset = df.iloc[:len(lstm_predictions_original)]

    # Convert 'Date' column to a NumPy array
    dates_array = df_subset['Date'].to_numpy()

    # Plot actual vs predicted prices in the original scale
    plt.plot(dates_array, y_test_original, label='Actual Prices', color='blue', linestyle='dashed')
    plt.plot(dates_array, lstm_predictions_original, label='Predicted Prices (LSTM)', color='red')

    # Set the x-axis to show only the months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.title('Actual vs Predicted Prices of {} Over Time'.format(ticker))
    plt.xlabel('Year of 2023')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def plot_SimpleRNN(df, y_test, rnn_predictions, ticker):
    min_value = df['Close'].min()
    max_value = df['Close'].max()

    feature_index = 0
    rnn_predictions_original = rnn_predictions[:, feature_index] * (max_value - min_value) + min_value
    y_test_original = y_test[:, feature_index] * (max_value - min_value) + min_value
    df['Date'] = pd.to_datetime(df['Date'])

    df_subset = df.iloc[:len(rnn_predictions_original)]

    # Convert 'Date' column to a NumPy array
    dates_array = df_subset['Date'].to_numpy()

    # Plot actual vs predicted prices in the original scale for Simple RNN
    plt.plot(dates_array, y_test_original, label='Actual Prices', color='blue', linestyle='dashed')
    plt.plot(dates_array, rnn_predictions_original, label='Predicted Prices (Simple RNN)', color='green')

    # Set the x-axis to show only the months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.title('Actual vs Predicted Prices of {} Over Time'.format(ticker))
    plt.xlabel('Year of 2023')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_cnn(df, y_test, cnn_predictions, ticker):
    min_value = df['Close'].min()
    max_value = df['Close'].max()

    feature_index = 0
    cnn_predictions_original = cnn_predictions[:, feature_index] * (max_value - min_value) + min_value
    y_test_original = y_test[:, feature_index] * (max_value - min_value) + min_value
    df['Date'] = pd.to_datetime(df['Date'])

    df_subset = df.iloc[:len(cnn_predictions_original)]

    # Convert 'Date' column to a NumPy array
    dates_array = df_subset['Date'].to_numpy()

    # Plot actual vs predicted prices in the original scale for CNN
    plt.plot(dates_array, y_test_original, label='Actual Prices', color='blue', linestyle='dashed')
    plt.plot(dates_array, cnn_predictions_original, label='Predicted Prices (CNN)', color='purple')

    # Set the x-axis to show only the months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.title('Actual vs Predicted Prices of {} Over Time'.format(ticker))
    plt.xlabel('Year of 2023')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
