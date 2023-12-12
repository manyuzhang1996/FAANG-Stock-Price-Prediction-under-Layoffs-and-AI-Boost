import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn

# Process the text: lower letters, remove stop words, remove punctuation, lemmanization
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_historical_dataset():
    df = pd.read_csv('NEWS_YAHOO_stock_prediction.csv', index_col = 0)
    df['combined_text'] = df['title'] + ' ' + df['content']
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    # VADER
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score_sia'] = df['combined_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    # TextBlob
    df['sentiment_score_textblob'] = df['combined_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # Afinn
    afinn = Afinn()
    df['sentiment_score_afinn'] = df['combined_text'].apply(lambda x: afinn.score(x))
    return df

def load_scrapped_dataset():
    df = pd.read_csv('news.csv')
    df['title'] = df['title'].apply(preprocess_text)
    return df