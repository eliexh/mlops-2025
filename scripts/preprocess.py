# scripts/preprocess.py
import argparse
import pandas as pd

def build_parser():
    p = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    p.add_argument("--input", required=True, help="Path to raw CSV")
    p.add_argument("--output", required=True, help="Path to save cleaned CSV")
    return p

def main():
    args = build_parser().parse_args()
    
    df = pd.read_csv(args.input)
    
    # Drop 'Cabin' (and any other unneeded columns if desired)
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
    
    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    
    df.to_csv(args.output, index=False)
    print(f"Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()


