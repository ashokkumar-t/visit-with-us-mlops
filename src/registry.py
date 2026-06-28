"""
registry.py 

Uploads trained model artifacts to the Hugging Face Model Hub.
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from config import (
    MODEL_DIR,
    MODEL_NAME,
    PREPROCESSOR_NAME,
    MODEL_REPO,
)

############################################################
# Authentication
############################################################

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN environment variable is not set."
    )

############################################################
# Hugging Face API
############################################################

api = HfApi(token=HF_TOKEN)

############################################################
# Create Repository
############################################################

def create_model_repo():

    try:

        create_repo(

            repo_id=MODEL_REPO,

            repo_type="model",

            token=HF_TOKEN,

            exist_ok=True

        )

        print("Model repository ready.")

    except Exception as e:

        print(e)

############################################################
# Upload File
############################################################

def upload(filepath):

    filepath = Path(filepath)

    print(f"Uploading {filepath.name}...")

    api.upload_file(

        path_or_fileobj=str(filepath),

        path_in_repo=filepath.name,

        repo_id=MODEL_REPO,

        repo_type="model"

    )

    print(f"{filepath.name} uploaded successfully.")

############################################################
# Main
############################################################

def main():

    print("=" * 60)

    print("MODEL REGISTRATION")

    print("=" * 60)

    create_model_repo()

    ########################################################

    model_file = MODEL_DIR / MODEL_NAME

    preprocessor_file = MODEL_DIR / PREPROCESSOR_NAME

    ########################################################

    if not model_file.exists():

        raise FileNotFoundError(model_file)

    if not preprocessor_file.exists():

        raise FileNotFoundError(preprocessor_file)

    ########################################################

    upload(model_file)

    upload(preprocessor_file)

    ########################################################

    print()

    print("=" * 60)

    print("Model Registration Completed Successfully")

    print("=" * 60)


if __name__ == "__main__":

    main()