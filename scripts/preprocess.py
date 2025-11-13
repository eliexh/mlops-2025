# scripts/preprocess.py
import sys
import os
import argparse
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.package.preprocessing.preprocessor import TitanicPreprocessor


def build_parser():
    p = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    p.add_argument("--input", required=True, help="Path to raw CSV")
    p.add_argument("--output", required=True, help="Path to save cleaned CSV")
    return p

def main():
    args = build_parser().parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Use the class
    preprocessor = TitanicPreprocessor()
    df_clean = preprocessor.process(df)

    # Save cleaned data
    df_clean.to_csv(args.output, index=False)
    print(f"✅ Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()
