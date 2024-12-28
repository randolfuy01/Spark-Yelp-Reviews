"""
File: regression.py

Description:
    This script is designed to load, preprocess, and classify tokenized text data 
    using machine learning models. It includes implementations for Random Forest 
    and Logistic Regression classifiers, with functionality to handle fixed-length 
    token sequences. The script leverages both PySpark for data processing and 
    scikit-learn for model training and evaluation.

Structure:
    - Imports: Standard, PySpark, and scikit-learn imports for data handling and ML tasks.
    - Functions:
        1. loader: Loads and preprocesses the dataset from a specified path, 
                   samples data by class, and splits it into train and test sets.
        2. preprocess_tokens: Converts tokenized text into fixed-length sequences 
                              by padding or truncating to a specified length.
        3. train_random_forest: Trains a Random Forest classifier on the processed data 
                                and evaluates its performance.
        4. train_logistic_regression: Trains a Logistic Regression model on the processed 
                                      data and evaluates its performance.
    - Main Function: Calls the loader and training functions to execute the full pipeline.

Usage:
    - Configure the dataset path, sample size, and token length in the `main` function.
    - Run the script to train both Random Forest and Logistic Regression models 
      and log their performance metrics.

Logging:
    - The script uses Python's logging module to track progress, report model metrics, 
      and ensure transparency during execution.

Requirements:
    - PySpark
    - scikit-learn
    - pandas
    - numpy

How to Run:
    Execute the script as follows:
        python model_training.py
"""

# Standard imports
import pandas as pd
import numpy as np
import logging

# Spark data processing imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def loader(load_path: str, sample_size: int) -> tuple:
    """
    Load data from a specified path, preprocess it, and split it into train and test sets.

    Parameters:
        load_path (str): Path to the dataset in parquet format.
        sample_size (int): Number of samples to retrieve for each star rating.

    Returns:
        tuple: A tuple containing the training and testing datasets as pandas DataFrames.
    """
    logger.info(f"Loading data from {load_path}")

    spark = (
        SparkSession.builder.appName("Data Loader")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "4g")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    loaded_data = spark.read.parquet(load_path)
    loaded_data = loaded_data.dropna(how="any")

    # Load tokenized text and stars only
    loaded_data = loaded_data.select(
        col("tokenized_text").alias("tokenized"), col("stars").alias("stars")
    )

    loaded_data.show(3)

    sampled_dataframe = []

    for star in range(1, 6):
        filtered_data = loaded_data.filter(col("stars") == star)
        sampled_data = filtered_data.orderBy(col("stars").desc()).limit(sample_size)
        sampled_dataframe.append(sampled_data.toPandas())

    # Combine all sampled data into a single Pandas DataFrame
    pandas_data = pd.concat(sampled_dataframe, ignore_index=True)

    # Split data into train and test sets
    train, test = train_test_split(pandas_data, test_size=0.2, random_state=42)
    return train, test


def preprocess_tokens(data: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """
    Preprocess the tokenized column to ensure all features are of fixed length.

    Parameters:
        data (pd.DataFrame): Input dataset containing a "tokenized" column.
        max_length (int): Maximum length for tokenized sequences.

    Returns:
        pd.DataFrame: Preprocessed dataset with fixed-length tokenized features.
    """

    # Pad or truncate tokens to a fixed length
    def pad_or_truncate(tokens):
        tokens = tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))

    # Apply the transformation
    data["tokenized"] = data["tokenized"].apply(pad_or_truncate)
    return data


def train_random_forest(train: pd.DataFrame, test: pd.DataFrame, max_length: int = 100):
    """
    Train a Random Forest Classifier on the tokenized dataset and evaluate its performance.

    Parameters:
        train (pd.DataFrame): Training dataset with "tokenized" and "stars" columns.
        test (pd.DataFrame): Testing dataset with "tokenized" and "stars" columns.
        max_length (int): Maximum length for tokenized sequences.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    logger.info("Starting preprocessing of tokens for fixed-length features...")

    # Preprocess tokens for fixed-length features
    train = preprocess_tokens(train, max_length)
    test = preprocess_tokens(test, max_length)

    logger.info("Finished preprocessing tokens.")

    # Extract features and labels
    logger.info("Extracting features and labels from the dataset...")
    X_train = np.array(train["tokenized"].to_list())
    y_train = train["stars"]

    X_test = np.array(test["tokenized"].to_list())
    y_test = test["stars"]
    logger.info(
        f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}"
    )

    # Initialize and train the Random Forest Classifier
    logger.info("Initializing and training the Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Random Forest model training complete.")

    # Make predictions
    logger.info("Making predictions on the test set...")
    predictions = model.predict(X_test)

    # Evaluate the model
    logger.info("Evaluating the model...")
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    logger.info(f"Model Accuracy: {accuracy:.2f}")
    logger.info("Classification Report:")
    logger.info(f"\n{report}")

    return model


def train_logistic_regression(
    train: pd.DataFrame, test: pd.DataFrame, max_length: int = 100
):
    """
    Train a Logistic Regression model on the tokenized dataset and evaluate its performance.

    Parameters:
        train (pd.DataFrame): Training dataset with "tokenized" and "stars" columns.
        test (pd.DataFrame): Testing dataset with "tokenized" and "stars" columns.
        max_length (int): Maximum length for tokenized sequences.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    logger.info("Starting preprocessing of tokens for fixed-length features...")

    # Preprocess tokens for fixed-length features
    train = preprocess_tokens(train, max_length)
    test = preprocess_tokens(test, max_length)

    logger.info("Finished preprocessing tokens.")

    # Extract features and labels
    logger.info("Extracting features and labels from the dataset...")
    X_train = np.array(train["tokenized"].to_list())
    y_train = train["stars"]

    X_test = np.array(test["tokenized"].to_list())
    y_test = test["stars"]
    logger.info(
        f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}"
    )

    # Initialize and train the Logistic Regression model
    logger.info("Initializing and training the Logistic Regression model...")
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression model training complete.")

    # Make predictions
    logger.info("Making predictions on the test set...")
    predictions = model.predict(X_test)

    # Evaluate the model
    logger.info("Evaluating the model...")
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    logger.info(f"Model Accuracy: {accuracy:.2f}")
    logger.info("Classification Report:")
    logger.info(f"\n{report}")

    return model


def main():
    """
    Main function to load the dataset, train models, and display results.
    """
    data_path = "../data/encoded_format/encoded_data.parquet"
    sample_size = 20000
    token_length = 200

    train, test = loader(data_path, sample_size)

    print("Training a Random Forest Classifier using sklearn...")
    forest_model = train_random_forest(train, test, max_length=token_length)

    print("Training a Logistic Regression Classifier using sklearn...")
    logistic_regression_model = train_logistic_regression(
        train, test, max_length=token_length
    )


if __name__ == "__main__":
    main()
