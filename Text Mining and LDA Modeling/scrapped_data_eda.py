import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

def scrapped_data_visualization(df):
    # 1) Group by company
    ticker_counts = df['related_tickers'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(ticker_counts, labels=ticker_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of News by Company')
    plt.show()

    # 2) Group by publisher
    publisher_counts = df['publisher'].value_counts()
    plt.figure(figsize=(15, 8))
    publisher_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Number of News Articles by {len(df['publisher'].unique())} Publishers")
    plt.xlabel('Publisher')
    plt.ylabel('Number of News Articles')
    plt.show()

    # 3）Overall wordcloud
    all_titles = ' '.join(df['title'])
    wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(all_titles)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # 4) Word Cloud stratified by company
    grouped_titles = df.groupby('related_tickers')['title'].apply(lambda x: ' '.join(x)).reset_index()
    wordclouds = defaultdict(WordCloud)
    for index, row in grouped_titles.iterrows():
        company = row['related_tickers']
        titles = row['title']
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(titles)
        wordclouds[company] = wordcloud
    for company, wordcloud in wordclouds.items():
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {company}')
        plt.axis('off')
        plt.show()