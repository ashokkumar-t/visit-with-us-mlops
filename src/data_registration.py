from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="visit-with-us-mlops/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id="itisashokkumar/visit-with-us-dataset",
    repo_type="dataset",
)

print("Dataset uploaded successfully!")


api.upload_file(
    path_or_fileobj="visit-with-us-mlops/data/train.csv",
    path_in_repo="train.csv",
    repo_id="itisashokkumar/visit-with-us-dataset",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="visit-with-us-mlops/data/test.csv",
    path_in_repo="test.csv",
    repo_id="itisashokkumar/visit-with-us-dataset",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="visit-with-us-mlops/data/tourism_cleaned.csv",
    path_in_repo="tourism_cleaned.csv",
    repo_id="itisashokkumar/visit-with-us-dataset",
    repo_type="dataset",
)