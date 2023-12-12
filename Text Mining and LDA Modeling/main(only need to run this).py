from data_processing import preprocess_text, load_historical_dataset, load_scrapped_dataset
from sentiment_score_comparing import compared_sentiment_scores
from scrapped_data_eda import scrapped_data_visualization
from lda_modeling import calculate_coherence, lda_modeling_experiment, optimal_lda_modeling, visualize_lda_topics

def main():
    # Load and process historical news data and scrapped news data
    historical_df = load_historical_dataset()
    new_df = load_scrapped_dataset()

    # Compare sentiment scores alignment with historic stock price to find the optimal sentiment scoring method
    compared_sentiment_scores(historical_df)

    # EDA on scrapped news data
    scrapped_data_visualization(new_df)

    # LDA topic modeling to find the optimal number of topics
    corpus, optimal_topics, dictionary = lda_modeling_experiment(new_df)

    # Get optimal LDA topics and keyword distribution
    lda_model = optimal_lda_modeling(corpus, optimal_topics, dictionary)

    # Visualize the LDA topics modeling results with word cloud
    visualize_lda_topics(lda_model, optimal_topics)


if __name__ == "__main__":
    main()