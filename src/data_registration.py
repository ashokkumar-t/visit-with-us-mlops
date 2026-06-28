"""
Register raw dataset to Hugging Face 
"""

from huggingface_hub import HfApi
from pathlib import Path
from config import DATASET_REPO, HF_TOKEN

RAW_FILE = Path("data/tourism.csv")
print("Current Directory:", Path.cwd())
print("Dataset Path:", RAW_FILE.resolve())

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE.resolve()}"
    )

def main():

    api = HfApi(token=HF_TOKEN)

    api.upload_file(
        path_or_fileobj=str(RAW_FILE),
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    print("Dataset uploaded successfully.")


if __name__ == "__main__":
    main()