@echo off
REM ===============================
REM Run full Titanic ML pipeline
REM ===============================

REM 1. Preprocess train and test
uv run python scripts/preprocess.py --input data/raw/train.csv --output data/processed/train_clean.csv
uv run python scripts/preprocess.py --input data/raw/test.csv --output data/processed/test_clean.csv

REM 2. Feature engineering
uv run python scripts/featurize.py --input data/processed/train_clean.csv --output data/processed/train_features.csv
uv run python scripts/featurize.py --input data/processed/test_clean.csv --output data/processed/test_features.csv

REM 3. Train model
uv run python scripts/train.py --input data/processed/train_features.csv --output models/logreg_model.pkl

REM 4. Evaluate
uv run python scripts/evaluate.py --model models/logreg_model.pkl --input data/processed/train_features.csv --output data/processed/metrics.json

REM 5. Predict on test set
uv run python scripts/preprocess.py --input data/raw/test.csv --output data/processed/test_clean.csv
uv run python scripts/featurize.py --input data/processed/test_clean.csv --output data/processed/test_features.csv
uv run python scripts/predict.py --model models/logreg_model.pkl --input data/processed/test_features.csv --output data/processed/test_predictions.csv

echo ===============================
echo Pipeline completed successfully!
echo ===============================
pause
