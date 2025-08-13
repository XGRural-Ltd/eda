from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Carregue os objetos exportados
rf = joblib.load('random_forest_model.pkl')
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
scaler_norm = joblib.load('scaler_norm.pkl')
selected_features = joblib.load('selected_features.pkl')

class InputData(BaseModel):
    features: dict  # Ex: {"popularity": 50, ...}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.features[f] for f in selected_features]])
    X = scaler.transform(X)
    X = scaler_norm.transform(X)
    pred = rf.predict(X)
    label = le.inverse_transform(pred)[0]
    return {"genre": label}
