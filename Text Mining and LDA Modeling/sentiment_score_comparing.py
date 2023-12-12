import pandas as pd
import matplotlib.pyplot as plt

def compared_sentiment_scores(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # daily stock close price vs. sentiment
    print("Daily Stock Close Price vs. Yearly Average Sentiment")
    # 1) stock price
    df.sort_index(inplace=True)
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Adj Close Price', color='tab:blue')
    ax1.plot(df.index, df['Adj Close'], color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Stock Close Price Over Time')
    plt.xlim(left=pd.to_datetime('2011-07-01'))
    fig.tight_layout()
    plt.show()

    # 2) sia sentiment
    df['Year'] = df.index.year
    yearly_sentiment = df.groupby('Year')['sentiment_score_sia'].mean()
    plt.figure(figsize=(18, 6))
    plt.plot(yearly_sentiment.index, yearly_sentiment, marker='o', linestyle='-', color='tab:red')
    plt.title('Yearly Average Sentiment Score - SIA')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    plt.show()

    # 3) textblob sentiment
    df['Year'] = df.index.year
    yearly_sentiment = df.groupby('Year')['sentiment_score_textblob'].mean()
    plt.figure(figsize=(18, 6))
    plt.plot(yearly_sentiment.index, yearly_sentiment, marker='o', linestyle='-', color='tab:red')
    plt.title('Yearly Average Sentiment Score - TextBlob')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    plt.show()

    # 4) afinn sentiment
    df['Year'] = df.index.year
    yearly_sentiment = df.groupby('Year')['sentiment_score_afinn'].mean()
    plt.figure(figsize=(18, 6))
    plt.plot(yearly_sentiment.index, yearly_sentiment, marker='o', linestyle='-', color='tab:red')
    plt.title('Yearly Average Sentiment Score - Afinn')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    plt.show()