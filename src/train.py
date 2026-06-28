"""
train.py

Model Training Pipeline 
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from pathlib import Path

from datasets import load_dataset

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import build_preprocessor
from model_factory import (
    get_models,
    get_parameter_grid,
)

from config import (
    DATASET_REPO,
    TRAIN_FILE,
    TARGET_COLUMN,
    MODEL_DIR,
    REPORT_DIR,
    MODEL_NAME,
    PREPROCESSOR_NAME,
    RANDOM_STATE,
    CV_FOLDS,
)


############################################################
# Load Dataset
############################################################

def load_training_data():

    dataset = load_dataset(DATASET_REPO)

    df = dataset["train"].to_pandas()

    if TRAIN_FILE in dataset.keys():
        df = dataset[TRAIN_FILE].to_pandas()

    return df


############################################################
# Train
############################################################

def train():

    print("=" * 60)
    print("Loading Training Dataset")
    print("=" * 60)

    df = load_training_data()

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    ########################################################

    print("Building Preprocessor...")

    preprocessor = build_preprocessor(X)

    ########################################################

    models = get_models()

    params = get_parameter_grid()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_model = None

    best_pipeline = None

    best_score = -1

    results = []

    ########################################################

    for name, model in models.items():

        print()

        print("=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        grid = GridSearchCV(

            estimator=pipeline,

            param_grid=params[name],

            scoring="f1",

            cv=cv,

            n_jobs=-1,

            verbose=1,

        )

        grid.fit(X, y)

        pred = grid.predict(X)

        accuracy = accuracy_score(y, pred)

        precision = precision_score(y, pred)

        recall = recall_score(y, pred)

        f1 = f1_score(y, pred)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "Best Parameters": grid.best_params_

        })

        if f1 > best_score:

            best_score = f1

            best_model = grid.best_estimator_

            best_pipeline = grid.best_estimator_

    ########################################################

    MODEL_DIR.mkdir(exist_ok=True)

    REPORT_DIR.mkdir(exist_ok=True)

    ########################################################

    print()

    print("=" * 60)
    print("Saving Best Model")
    print("=" * 60)

    joblib.dump(

        best_pipeline,

        MODEL_DIR / MODEL_NAME

    )

    ########################################################

    joblib.dump(

        preprocessor,

        MODEL_DIR / PREPROCESSOR_NAME

    )

    ########################################################

    results_df = pd.DataFrame(results)

    results_df.to_csv(

        REPORT_DIR / "model_results.csv",

        index=False

    )

    ########################################################

    print()

    print("=" * 60)

    print("Training Completed")

    print(f"Best F1 Score : {best_score:.4f}")

    print("=" * 60)


############################################################

if __name__ == "__main__":

    train()