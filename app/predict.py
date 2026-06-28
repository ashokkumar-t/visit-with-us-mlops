import pandas as pd
from model_loader import load_pipeline
_pipeline=load_pipeline()
def predict_customer(data):
    df=pd.DataFrame([data])
    pred=_pipeline.predict(df)[0]
    prob=_pipeline.predict_proba(df)[0][1]
    return ('Purchased' if pred==1 else 'Not Purchased'), prob
