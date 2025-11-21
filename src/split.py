import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COL = "target"

def main(input_csv: str, out_dir: str, test_size: float):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    train = X_train.copy()
    train[TARGET_COL] = y_train.values
    test = X_test.copy()
    test[TARGET_COL] = y_test.values

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"[OK] Guardado train en: {train_path} (shape={train.shape})")
    print(f"[OK] Guardado test  en: {test_path} (shape={test.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv",
                        default="data/processed/digits_clean.csv", help="CSV de features")
    parser.add_argument("--outdir", default="data/processed",
                        help="Carpeta de salida para train/test")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proporción de test")
    args = parser.parse_args()
    main(args.input_csv, args.outdir, args.test_size)
