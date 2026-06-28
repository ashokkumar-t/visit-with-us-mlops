"""
Application Configuration
"""

import os

###########################################################
# Hugging Face
###########################################################

HF_MODEL_REPO = "itisashokkumar/visit-with-us-models"

MODEL_FILENAME = "best_model.pkl"

###########################################################
# Streamlit
###########################################################

APP_TITLE = "Visit With Us"

APP_SUBTITLE = "Wellness Tourism Package Purchase Prediction"

VERSION = "1.0.0"

AUTHOR = "Ashokkumar T"

###########################################################
# Prediction Labels
###########################################################

PREDICTION_LABELS = {
    0: "Not Likely to Purchase",
    1: "Likely to Purchase"
}