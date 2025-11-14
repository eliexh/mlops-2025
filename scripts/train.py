# scripts/train.py
import sys
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.package.models.logistic_model import LogisticModel
from src.package.models.random_forest_model import RandomForestModel
from src.package.models.xgboost_model import XGBoostModel


def build_parser():
    p = argparse.ArgumentParser(description="Train a model")
    p.add_argument("--input", required=True, help="Path to feature CSV")
    p.add_argument("--output", required=True, help="Path to save trained model (.pkl)")
    p.add_argument("--model", choices=["logreg", "rf", "xgb"], default="logreg",
                   help="Which model to train")
    return p


def main():
    args = build_parser().parse_args()

    # Load dataset
    df = pd.read_csv(args.input)

    # Split into X, y
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pick model
    if args.model == "logreg":
        print("🟧 Training Logistic Regression...")
        model = LogisticModel()
    elif args.model == "rf":
        print("🌲 Training RandomForest...")
        model = RandomForestModel()
    else:
        print("Training XGBoost...")
        model = XGBoostModel()  

    # TRAIN THE MODEL (you were missing this!!!)
    model.train(X_train, y_train)

    # Evaluate
    acc = model.evaluate(X_test, y_test)
    print(f"✅ Model trained. Accuracy: {acc:.3f}")

    # Save the model using its own save() method
    model.save(args.output)
    print(f"💾 Model saved to {args.output}")


if __name__ == "__main__":
    main()


