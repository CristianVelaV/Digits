import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TARGET_COL = "target"

def main(train_csv: str, test_csv: str, model_path: str, max_iter: int):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(dummy, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")
    print(f"[MÉTRICA] Accuracy test: {acc:.4f}")

    metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
    cm_path = os.path.join(os.path.dirname(model_path), "confusion_matrix.png")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot(values_format="d")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    metrics = {
        "accuracy": acc,
    }   
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Métricas guardadas en: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv",
                        help="CSV de entrenamiento")
    parser.add_argument("--test", default="data/processed/test.csv",
                        help="CSV de prueba")
    parser.add_argument("--out", dest="model_path", default="models/model.pkl",
                        help="Ruta del modelo")
    parser.add_argument("--max_iter", type=int, default=500,
                        help="Iteraciones máximas del solver")
    args = parser.parse_args()
    main(args.train, args.test, args.model_path, args.max_iter)
