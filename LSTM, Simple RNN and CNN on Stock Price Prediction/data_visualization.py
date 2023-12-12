# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:14:55 2023

@author: yunhui
"""

from dataset import get_stock_data
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def data_visualization():
    # Load stock data
    df = get_stock_data()

    # Plotting closing prices over time for all tickers
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Close', hue='Ticker', data=df)
    plt.title('Closing Prices Over Time for All Tickers')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend(loc='upper left')
    plt.show()

    # Plotting the sum of volume for each ticker as histograms
    sum_volume_per_ticker = df.groupby('Ticker')['Volume'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Ticker', y='Volume', data=sum_volume_per_ticker)
    plt.title('Total Volume for Each Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Sum of Volume')
    plt.show()

    # Calculate the correlation matrix
    df['Date'] = df['Date'].apply(lambda x: datetime.datetime.combine(x, datetime.time()).timestamp())

    # Calculate the correlation matrix
    correlation_matrix = df.groupby('Ticker').corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix['Close'].unstack(level=0), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap for Close Prices by Ticker')
    plt.show()
