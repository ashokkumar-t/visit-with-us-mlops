"""
predict.py

Prediction service for the Streamlit application.
"""

import pandas as pd

from model_loader import load_model
from config import PREDICTION_LABELS


##############################################################
# Load model only once
##############################################################

model = load_model()


##############################################################
# Validate Input
##############################################################

def validate_input(df: pd.DataFrame):
    """
    Validate user input.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    bool
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    return True


##############################################################
# Prediction
##############################################################

def predict(df: pd.DataFrame):
    """
    Predict whether a customer is likely
    to purchase the tourism package.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
    """

    validate_input(df)

    ##########################################################

    prediction = model.predict(df)[0]

    ##########################################################
    # Probability (if available)
    ##########################################################

    probability = None

    if hasattr(model, "predict_proba"):

        try:

            probability = float(

                model.predict_proba(df)[0][1]

            )

        except Exception:

            probability = None

    ##########################################################

    return {

        "prediction": int(prediction),

        "label": PREDICTION_LABELS[int(prediction)],

        "probability": probability

    }


##############################################################
# Batch Prediction
##############################################################

def predict_batch(df: pd.DataFrame):
    """
    Predict multiple customers.
    """

    validate_input(df)

    predictions = model.predict(df)

    results = df.copy()

    results["Prediction"] = predictions

    results["Result"] = results["Prediction"].map(
        PREDICTION_LABELS
    )

    return results


##############################################################
# Health Check
##############################################################

def health():
    """
    Check if prediction service is ready.
    """

    try:

        _ = model

        return True

    except Exception:

        return False


##############################################################

if __name__ == "__main__":

    print("Prediction Service Ready")