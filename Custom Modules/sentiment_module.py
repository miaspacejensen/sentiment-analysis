import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def dataset_prep(data, sample_size):
    # Sample the data
    if sample_size > 0: data = data.sample(n=sample_size)

    # define X and y sets
    y = data["Score"].to_frame()
    X = data.drop(columns=["Score"])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    return X_train, X_test, y_train, y_test

def clean_data(X):
    # drop ID columns
    X = X.drop(columns=["Unnamed: 0", "Id"])

    # Remove duplicate rows (i.e. users who have reviewed the same product with the same review details)
    X = X.drop_duplicates()

    # drop all except Text
    X = X[["Text"]]

    print(X.shape)

    return X

def preprocess_text(text):
    # processing steps to consider
    # - reduce repeated letters e.g. yeeeeeees -> yes
    # - remove links
    # - remove frequent words
    # - remove rare words

    # tokenisation
    tokens = word_tokenize(text.lower())

    # remove stopwords from tokens
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # lemmatisation of tokens
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(t) for t in tokens]

    # remove non-alphanumeric characters from tokens
    # remove empty tokens
    tokens_alphanum = []
    for token in tokens:
        token_alphanum = re.sub(r'[^0-9a-zA-Z\s]+', '', token)
        if len(token_alphanum) > 0:
            tokens_alphanum.append(token_alphanum)

    # consolidate all tokens into string
    processed_text = ' '.join(tokens_alphanum)

    return processed_text

def get_sentiment(text):
    analyser = SentimentIntensityAnalyzer()
    scores = analyser.polarity_scores(text) 
    if scores['pos'] > 0:
        sentiment = 1
    else:
        sentiment = 0
    return sentiment

def sentiment_analysis(X, y):
    
    # cleaning
    print("\nCleaning...")
    X = clean_data(X)

    # preprocessing
    print("\nPreprocessing...")
    X["Text"] = X["Text"].apply(preprocess_text)

    # sentiment analysis
    print("\nAnalysing...")
    y["ScoreBinary"] = np.where(y>=3, 1, 0)
    X["Sentiment"] = X["Text"].apply(get_sentiment)

    # combine datasets
    data = pd.concat([X, y], axis=1)

    # results
    print("\n-----Results-----")
    conf_mat = confusion_matrix(data['ScoreBinary'], data['Sentiment'])
    class_report = classification_report(data['ScoreBinary'], data['Sentiment'])
    accuracy = accuracy_score(data['ScoreBinary'], data['Sentiment'])
    print(
        "\nConfusion Matrix:\n", conf_mat,
        "\n\nClassification Report:\n", class_report,
        "\nAccuracy =", f"{accuracy*100}%"
    )

    print("Complete")
    
    return data, X, y, conf_mat, class_report, accuracy