from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# -------------------------
# Cargar modelo
# -------------------------
MODEL_PATH = "models/model.pkl"   # cambia el nombre si tu modelo es otro
model = joblib.load(MODEL_PATH)

# -------------------------
# Definir API
# -------------------------
app = FastAPI(
    title="ML Model API",
    description="API para clasificar dibujos en digitos",
    version="1.0"
)

# -------------------------
# Esquema de entrada
# -------------------------
class InputData(BaseModel):
    values: list  # una fila de features


@app.get("/")
def root():
    return {"status": "API funcionando correctamente"}


@app.post("/predict")
def predict(data: InputData):
    
    # Convertir a array numpy del shape (1, n_features)
    X = np.array(data.values).reshape(1, -1)

    # Hacer predicción
    pred = model.predict(X)[0]

    # Si el modelo tiene probability=True
    try:
        proba = model.predict_proba(X)[0].tolist()
    except:
        proba = None

    return {
        "prediction": pred,
        "probabilities": proba
    }