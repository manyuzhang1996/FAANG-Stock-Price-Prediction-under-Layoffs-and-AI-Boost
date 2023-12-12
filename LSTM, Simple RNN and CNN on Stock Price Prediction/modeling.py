# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:24:31 2023

@author: yunhui
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import numpy as np
from data_preprocessing import calculate_mape

def lstm_model(X_train, X_test, y_train, y_test, ticker):

  print("Training LSTM for {}:".format(ticker))
  # Define and compile the LSTM model
  lstm_model = Sequential()
  lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
  lstm_model.add(Dense(4, activation='linear'))
  lstm_model.compile(optimizer='adam', loss='mse')

  # Train the LSTM model
  lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

  # Make predictions on the test set
  lstm_predictions = lstm_model.predict(X_test)

  # Reshape predictions to match the shape of y_test
  lstm_predictions = lstm_predictions.reshape(y_test.shape)

  # Calculate Evaluation Metrics
  rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
  mape = calculate_mape(y_test, lstm_predictions)
  model = 'LSTM'
 

  print(f'Root Mean Squared Error (RMSE): {rmse}')
  print(f'Mean Absolute Percentage Error (MAPE): {mape}')

  return lstm_predictions, rmse, mape

def rnn_model(X_train, X_test, y_train, y_test, ticker):

  print("Training Simple RNN for {}:".format(ticker))
  # Define Simple RNN model
  rnn_model = Sequential()
  rnn_model.add(SimpleRNN(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
  rnn_model.add(Dense(4))
  rnn_model.compile(optimizer='adam', loss='mse')

  # Train RNN model
  rnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

  # Predict using RNN model
  rnn_predictions = rnn_model.predict(X_test)

  rmse = np.sqrt(mean_squared_error(y_test, rnn_predictions))
  mape = calculate_mape(y_test, rnn_predictions)
  model = 'Simple RNN'


  print(f'Root Mean Squared Error (RMSE): {rmse}')
  print(f'Mean Absolute Percentage Error (MAPE): {mape}')

  return rnn_predictions, rmse, mape

def cnn_model(X_train, X_test, y_train, y_test, ticker):

  print("Training CNN for {}:".format(ticker))
  # Reshape X_train and X_test for CNN
  X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
  X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

  # Define CNN model
  cnn_model = Sequential()
  cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
  cnn_model.add(MaxPooling1D(pool_size=2))
  cnn_model.add(Flatten())
  cnn_model.add(Dense(50, activation='relu'))
  cnn_model.add(Dense(4))
  cnn_model.compile(optimizer='adam', loss='mse')

  # Train CNN model
  cnn_model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_data=(X_test_cnn, y_test))

  # Predict using CNN model
  cnn_predictions = cnn_model.predict(X_test_cnn)

  # Evaluate CNN model
  rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))
  mape = calculate_mape(y_test, cnn_predictions)
  model = 'CNN'

  print(f'Root Mean Squared Error (RMSE): {rmse}')
  print(f'Mean Absolute Percentage Error (MAPE): {mape}')

  return cnn_predictions, rmse, mape