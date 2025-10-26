import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

# Add the data/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'src'))
from model import EmissionsPredictor
from data_processing import load_and_clean_data, prepare_features

def load_data():
    possible_paths = [
        "data/processed/emissions_cleaned.csv",
        "data/raw/epa_ghgrp_2021_2023_aggregate.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"📂 Loading dataset from {path} ...")
            df = pd.read_csv(path)
            print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns")
            break
    else:
        raise FileNotFoundError("❌ Could not find dataset in data/processed or data/raw")

    # Use the actual column names from the CSV
    df = df.dropna(subset=["state", "industry_sector", "reporting_year", "total_ghg_emissions_tonnes"])
    
    # Create the cleaned industry sector column
    df['industry_sector_clean'] = df['industry_sector'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Other')
    
    X = df[["state", "industry_sector_clean", "reporting_year"]].copy()
    y = df["total_ghg_emissions_tonnes"]
    return X, y

def main():
    X, y = load_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictor = EmissionsPredictor()

    # Preprocess and fit base model
    X_train_encoded = predictor.preprocess(X_train, fit=True)
    X_test_encoded = predictor.preprocess(X_test, fit=False)

    print("\n🚀 Starting GridSearchCV hyperparameter tuning...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        predictor.model,
        param_grid,
        cv=3,
        scoring="r2",
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train_encoded, y_train)
    best_model = grid_search.best_estimator_
    print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
    print(f"🏅 Best R2 Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    predictor.model = best_model
    y_pred = predictor.predict(X_test)
    metrics = predictor.evaluate(y_test, y_pred)
    print("\n📊 Final Test Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")

    # Retrain on full data
    print("\n🔁 Retraining best model on full dataset...")
    X_encoded = predictor.preprocess(X, fit=True)
    predictor.model.fit(X_encoded, y)

    # Save model
    predictor.save_model("data/models/trained_model.pkl")
    print("\n✅ Training complete! Model saved successfully.")

if __name__ == "__main__":
    main()
