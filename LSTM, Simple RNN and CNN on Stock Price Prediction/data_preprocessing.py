# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:20:20 2023

@author: yunhui
"""

from dataset import get_stock_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime


def preprocess_data(df):
    # Feature Selection
    features = df[['Open', 'High', 'Low', 'Adj Close']].values
    target = df['Close'].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Create sequences for time series data
    def create_sequences(data, sequence_length):
        sequences = []
        labels = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            label = data[i+sequence_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    # Choose sequence length
    sequence_length = 10
    X, y = create_sequences(scaled_features, sequence_length)

    # Set the splitting date: Data before 1/1/2023 will be used for training, and data after will be used for testing.
    split_date = datetime.datetime(2023, 1, 1).date()  # Convert to datetime.date

    # Split the sequences and labels into training and testing sets based on the date
    train_size = np.sum(df['Date'] < split_date)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    mask = y_true != 0  # Create a mask to identify non-zero values in y_true
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))[mask]) * 100
    return f"{mape:.2f}%"

