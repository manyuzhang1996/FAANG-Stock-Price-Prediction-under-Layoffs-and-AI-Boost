# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:31:01 2023

@author: yunhui
"""

from dataset import get_stock_data
from modeling import lstm_model, rnn_model, cnn_model
from data_preprocessing import preprocess_data, calculate_mape
from actual_vs_predicted_price_plot import plot_lstm, plot_SimpleRNN, plot_cnn
import io

df = get_stock_data()


def run_models_and_plot(ticker):

    print("----------- Running LSTM, Simple RNN and CNN on {} ticker ---------\n".format(ticker))

    df_ticker = df[df['Ticker'] == ticker]

    X_train, X_test, y_train, y_test = preprocess_data(df_ticker)

    lstm_predictions, lstm_rmse, lstm_mape = lstm_model(X_train, X_test, y_train, y_test, ticker)
    rnn_predictions, rnn_rmse, rnn_mape = rnn_model(X_train, X_test, y_train, y_test, ticker)
    cnn_predictions, cnn_rmse, cnn_mape = cnn_model(X_train, X_test, y_train, y_test, ticker)

    plot_lstm(df_ticker, y_test, lstm_predictions, ticker)
    plot_SimpleRNN(df_ticker, y_test, rnn_predictions, ticker)
    plot_cnn(df_ticker, y_test, cnn_predictions, ticker)

    return (
        "\nResult for {}\n".format(ticker) +
        f'LSTM_RMSE: {round(lstm_rmse, 2)}, \n'
        f'LSTM_MAPE: {lstm_mape}, \n'
        f'SimpleRNN_RMSE: {round(rnn_rmse, 2)}, \n'
        f'SimpleRNN_MAPE: {rnn_mape}, \n'
        f'CNN_RMSE: {round(cnn_rmse, 2)}, \n'
        f'CNN_MAPE: {cnn_mape}'
    )



