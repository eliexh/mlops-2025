@echo off
REM ==============================
REM Run the full Titanic pipeline
REM ==============================

REM Set Python from your virtual environment
set PYTHON=%~dp0\.venv\Scripts\python.exe

echo ✅ Step 1: Preprocessing
"%PYTHON%" scripts\preprocess.py --input data\raw\train.csv --output data\processed\train_clean.csv

echo ✅ Step 2: Feature Engineering
"%PYTHON%" scripts\featurize.py --input data\processed\train_clean.csv --output data\processed\train_features.csv

echo ✅ Step 3a: Train Logistic Regression
"%PYTHON%" scripts\train.py --input data\processed\train_features.csv --output src\package\models\logreg_model.pkl --model logreg

echo ✅ Step 3b: Train Random Forest
"%PYTHON%" scripts\train.py --input data\processed\train_features.csv --output src\package\models\rf_model.pkl --model rf

echo ✅ Step 3c: Train XGBoost
"%PYTHON%" scripts\train.py --input data\processed\train_features.csv --output src\package\models\xgb_model.pkl --model xgb

echo 🎉 Pipeline completed!
pause
