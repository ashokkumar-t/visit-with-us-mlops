"""
Common Utility Functions 
"""

import json
import joblib
from pathlib import Path


def save_model(model, filepath):
    """
    Save model using joblib.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)

    print(f"Model saved at {filepath}")


def load_model(filepath):
    """
    Load saved model.
    """

    model = joblib.load(filepath)

    return model


def save_json(data, filepath):
    """
    Save dictionary as JSON.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as fp:
        json.dump(data, fp, indent=4)

    print(f"JSON saved at {filepath}")


def load_json(filepath):
    """
    Load JSON file.
    """

    with open(filepath, "r") as fp:
        return json.load(fp)


def banner(message):
    """
    Print banner.
    """

    print("=" * 60)
    print(message)
    print("=" * 60)