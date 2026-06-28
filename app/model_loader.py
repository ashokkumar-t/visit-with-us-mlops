from huggingface_hub import hf_hub_download
import joblib
REPO_ID='itisashokkumar/visit-with-us-models'
def load_pipeline():
    path=hf_hub_download(repo_id=REPO_ID,filename='best_model.pkl',repo_type='model')
    return joblib.load(path)
