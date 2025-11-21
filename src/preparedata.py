import argparse
import os
import pandas as pd

REQUIRED_COLS = [
     "target"
]

def main(input_csv: str, output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)

    # Verificación de esquema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Chequeos simples
    if df.isna().sum().sum() > 0:
        # Imputación simple (ilustrativa para docencia)
        df = df.fillna(df.mean(numeric_only=True))

    # Remover duplicados
    df = df.drop_duplicates()

    df.to_csv(output_csv, index=False)
    print(f"[OK] Guardado limpio en: {output_csv} (shape={df.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv",
                        default="data/raw/digits.csv", help="CSV crudo")
    parser.add_argument("--out", dest="output_csv",
                        default="data/processed/digits_clean.csv", help="CSV limpio")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
