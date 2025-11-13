# scripts/train.py
import sys
import os
import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.package.models.logistic_model import LogisticModel

def build_parser():
    p = argparse.ArgumentParser(description="Train a Logistic Regression model")
    p.add_argument("--input", required=True, help="Path to feature CSV")
    p.add_argument("--output", required=True, help="Path to save trained model")
    return p

def main():
     args = build_parser().parse_args()
     df = pd.read_csv(args.input)

     # Split data
     X = df.drop(columns=["Survived"])
     y = df["Survived"]

     # Convert categorical variables to numeric
     for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])


     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Train model
     model = LogisticModel()
     model.train(X_train, y_train)

     # Evaluate model
     acc = model.evaluate(X_test, y_test)
     print(f"✅ Model trained. Accuracy: {acc:.3f}")

     # Save model
     model.save(args.output)
     print(f"💾 Model saved to {args.output}")


