import pandas as pd
import os
from helper_functions import preprocess_data, split_data, oversample_data, visualize_class_balance, train_and_evaluate_models

if __name__ == "__main__":
    # Load data
    default_data_path = 'final_stocks.csv'
    data_path = os.getenv('DATA_PATH', default_data_path)
    try:
        news_stocks = pd.read_csv(data_path, index_col=0)
    except FileNotFoundError:
        print("Error: Data file not found.")
        exit(1)

    # Preprocess data
    news_stocks = preprocess_data(news_stocks)
    # Split the data
    X_train, X_test, y_train, y_test = split_data(news_stocks)

    # Oversampling data
    X_train_resampled, y_train_resampled = oversample_data(X_train, y_train)

    # Visualize class balance
    visualize_class_balance(y_train_resampled)

    # Train and evaluate models
    summary_df = train_and_evaluate_models(X_train_resampled, X_test, y_train_resampled, y_test)

    print(summary_df)