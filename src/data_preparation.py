"""
Data Preparation Pipeline 
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import (
    DATASET_REPO,
    TRAIN_FILE,
    TEST_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    DATA_DIR,
    HF_TOKEN,
)

from data_utils import (
    remove_columns,
    save_csv,
)

from huggingface_hub import HfApi

import pandas as pd


def download_dataset():

    dataset = load_dataset(DATASET_REPO)

    df = dataset["train"].to_pandas()

    return df


def clean_dataset(df):

    # Remove CustomerID
    df = remove_columns(df, ["CustomerID"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


def split_dataset(df):

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train = X_train.copy()

    train[TARGET_COLUMN] = y_train.values

    test = X_test.copy()

    test[TARGET_COLUMN] = y_test.values

    return train, test


def upload_dataset(train_path, test_path):

    api = HfApi(token=HF_TOKEN)

    api.upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo=TRAIN_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo=TEST_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    print("Train/Test uploaded successfully.")


def main():

    print("=" * 60)
    print("Downloading Dataset")
    print("=" * 60)

    df = download_dataset()

    print(df.shape)

    print("=" * 60)
    print("Cleaning Dataset")
    print("=" * 60)

    df = clean_dataset(df)

    print(df.shape)

    train, test = split_dataset(df)

    train_path = DATA_DIR / TRAIN_FILE

    test_path = DATA_DIR / TEST_FILE

    save_csv(train, train_path)

    save_csv(test, test_path)

    upload_dataset(train_path, test_path)

    print("Data Preparation Completed.")


if __name__ == "__main__":
    main()