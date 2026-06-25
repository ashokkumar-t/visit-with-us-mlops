from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="visit-with-us-mlops/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id="itisashokkumar/visit-with-us-dataset",
    repo_type="dataset",
)

print("Dataset uploaded successfully!")