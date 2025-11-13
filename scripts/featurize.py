# scripts/featurize.py
import sys
import os
import argparse
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.package.features.features_computer import TitanicFeaturesComputer


def build_parser():
    p = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    p.add_argument("--input", required=True, help="Path to cleaned CSV")
    p.add_argument("--output", required=True, help="Path to save feature CSV")
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)

    # Use feature engineering class
    featurizer = TitanicFeaturesComputer()
    df_features = featurizer.compute(df)

    df_features.to_csv(args.output, index=False)
    print(f"✅ Feature-engineered data saved to {args.output}")
  

if __name__ == "__main__":
    main()

