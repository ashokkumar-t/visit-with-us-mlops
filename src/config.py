"""
Project Configuration

Centralized configuration file used across the entire project. 
"""

from pathlib import Path
import os

############################################################
# Project Paths
############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

############################################################
# Dataset
############################################################

HF_USERNAME = os.getenv("HF_USERNAME", "itisashokkumar")

DATASET_REPO = "itisashokkumar/visit-with-us-dataset"

MODEL_REPO = "itisashokkumar/visit-with-us-models"

RAW_DATASET = "tourism.csv"

TRAIN_FILE = "train.csv"

TEST_FILE = "test.csv"

############################################################
# Model
############################################################

MODEL_NAME = "best_model.pkl"

PREPROCESSOR_NAME = "preprocessor.pkl"

############################################################
# Training
############################################################

TARGET_COLUMN = "ProdTaken"

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

############################################################
# Hugging Face
############################################################

HF_TOKEN = os.getenv("HF_TOKEN")

############################################################
# Reports
############################################################

METRICS_FILE = REPORT_DIR / "metrics.json"

RESULT_FILE = REPORT_DIR / "model_results.csv"

############################################################
# Create folders automatically
############################################################

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)