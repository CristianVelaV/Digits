import argparse
import os 
import pandas as pd
from sklearn.datasets import load_digits

def main(output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    data = load_digits(as_frame=True)
    X = data.data.copy()
    y = data.target
    df = X.assign(target=y)
    df.to_csv(output_csv, index=False)
    print(f"Dataset crudo guardado en {output_csv} (shape={df.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/digits.csv",
                        help="Ruta de salida CSV crudo")
    args = parser.parse_args()
    main(args.out)