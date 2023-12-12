# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

warnings.filterwarnings("ignore")


# Data Preprocessing
def preprocess_data(data):
    # Drop unnecessary columns
    data.drop(['date', 'publish_time', 'title', 'publisher'], axis=1, inplace=True)

    # One-hot encoding for 'ticker' column
    one_hot = pd.get_dummies(data['ticker'])
    # Drop 'ticker' column
    data = data.drop('ticker', axis=1)

    # Labeling
    X = data.drop(['label'], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    final_df = pd.DataFrame(X, columns=data.drop(['label'], axis=1).columns)
    final_df = final_df.join(one_hot)
    final_df = final_df.join(data['label'])

    return final_df


# Split the data into train and test sets
def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


# Apply random oversampling
def oversample_data(X_train, y_train):
    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


# Data Exploration and Visualization that will not be shown in py file
def visualize_label_distribution(data):
    # Grouping the data by 'ticker' and counting the occurrences of each label
    label_counts_by_ticker = data.groupby(['ticker', 'label']).size().unstack(fill_value=0)

    # Plotting the raw counts of labels for each ticker
    plt.figure(figsize=(12, 6))
    label_counts_by_ticker.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Distribution of Labels by Ticker (Raw Counts)')
    plt.xlabel('Ticker')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Visualize class balance
def visualize_class_balance(y_train):
    class_counts = Counter(y_train)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Balance before Random Oversampling')
    plt.show()


# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.05, n_estimators=110, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=150, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, max_features='sqrt', random_state=42)
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })

    summary_df = pd.DataFrame(results)
    return summary_df
