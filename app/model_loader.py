"""
model_loader.py

Downloads and loads the latest trained model from
the Hugging Face Model Hub.
"""

import streamlit as st
import joblib

from huggingface_hub import hf_hub_download

from config import (
    HF_MODEL_REPO,
    MODEL_FILENAME,
)

###########################################################
# Load Model
###########################################################

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Downloads the latest model from Hugging Face
    and caches it for future predictions.
    """

    try:

        model_path = hf_hub_download(

            repo_id=HF_MODEL_REPO,

            filename=MODEL_FILENAME,

            repo_type="model"

        )

        model = joblib.load(model_path)

        return model

    except Exception as e:

        raise RuntimeError(

            f"Unable to load model from Hugging Face.\n\n{e}"

        )


###########################################################
# Model Information
###########################################################

def model_information(model):
    """
    Returns model information.
    """

    info = {

        "Model Type": type(model).__name__

    }

    if hasattr(model, "named_steps"):

        info["Pipeline"] = list(

            model.named_steps.keys()

        )

    return info


###########################################################
# Health Check
###########################################################

def health_check():

    try:

        model = load_model()

        return True

    except Exception:

        return False


###########################################################

if __name__ == "__main__":

    print("Loading Model...")

    model = load_model()

    print()

    print(model_information(model))