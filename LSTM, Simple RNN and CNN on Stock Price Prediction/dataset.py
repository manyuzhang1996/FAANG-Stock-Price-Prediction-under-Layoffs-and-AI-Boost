# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:58:10 2023

@author: yunhui
"""

import yfinance as yf
import datetime
import pandas as pd


def get_stock_data():
    # List of tickers
    tickers = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

    # Start and end dates
    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2023, 11, 17)

    df = pd.DataFrame()

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date)

        # Concatenate data to the DataFrame
        df = pd.concat([df, historical_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Stock Splits', 'Dividends']]])

    df['Ticker'] = df.groupby(level=0).cumcount().map({i: tickers[i] for i in range(len(tickers))})
    df['Date'] = pd.to_datetime(df.index).date
    df['Adj Close'] = df['Close'] - df['Stock Splits'] - df['Dividends']

    df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

    df = df.reset_index(drop=True)
    df = df.sort_values(by='Date')

    return df