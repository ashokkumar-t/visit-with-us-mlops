"""
evaluate.py

Evaluate the trained model using the test dataset. 
"""

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    DATASET_REPO,
    TARGET_COLUMN,
    MODEL_DIR,
    REPORT_DIR,
    MODEL_NAME,
)

############################################################
# Load Test Dataset
############################################################

def load_test_data():

    dataset = load_dataset(DATASET_REPO)

    # Try loading test split
    if "test" in dataset.keys():
        df = dataset["test"].to_pandas()
    else:
        # fallback if uploaded as csv
        df = pd.read_csv("data/test.csv")

    return df


############################################################
# Evaluation
############################################################

def evaluate():

    print("=" * 60)
    print("Loading Model")
    print("=" * 60)

    model = joblib.load(MODEL_DIR / MODEL_NAME)

    print("=" * 60)
    print("Loading Test Dataset")
    print("=" * 60)

    df = load_test_data()

    X_test = df.drop(columns=[TARGET_COLUMN])

    y_test = df[TARGET_COLUMN]

    ########################################################

    print("Predicting...")

    predictions = model.predict(X_test)

    ########################################################

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    ########################################################

    metrics = {

        "Accuracy": round(accuracy,4),

        "Precision": round(precision,4),

        "Recall": round(recall,4),

        "F1 Score": round(f1,4)

    }

    ########################################################

    REPORT_DIR.mkdir(exist_ok=True)

    ########################################################

    with open(REPORT_DIR / "metrics.json","w") as fp:

        json.dump(metrics, fp, indent=4)

    ########################################################

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    ########################################################

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png"
    )

    plt.close()

    ########################################################

    print()

    print("="*60)

    print("Evaluation Results")

    print("="*60)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print()

    print("Reports saved to reports/")

    print("="*60)


############################################################

if __name__ == "__main__":

    evaluate()