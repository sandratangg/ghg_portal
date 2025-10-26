#!/usr/bin/env python3

"""
Complete pipeline to train the machine learning model and prepare the application.
Run this script before launching the Streamlit app.
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'src'))

from data_processing import load_and_clean_data, prepare_features
from model import EmissionsPredictor

def main():
    print("🌍 GHG Emissions Portal - Model Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load and clean data
    
    try:
        df = load_and_clean_data('data/raw/epa_ghgrp_2021_2023_aggregate.csv')
        print(f"Data loaded successfully: {df.shape[0]} facilities, {df.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Prepare features
    try:
        X, y = prepare_features(df)
        print(f"Features prepared: {X.shape[1]} features for modeling")
        print(f"Target variable (emissions) range: {y.min():,.0f} - {y.max():,.0f} metric tons")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return
    
    # Step 3: Train Random Forest model
    try:
        rf_predictor = EmissionsPredictor(model_type='random_forest')
        rf_results = rf_predictor.train(X, y)
        rf_predictor.save_model('data/models/random_forest_model.pkl')
        print("Random Forest model trained and saved")
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        return
    
    # Step 4: Train Linear Regression baseline
    try:
        lr_predictor = EmissionsPredictor(model_type='linear')
        lr_results = lr_predictor.train(X, y)
        lr_predictor.save_model('data/models/linear_model.pkl')
        print("Linear Regression model trained and saved")
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
        return
    
    # Step 5: Model comparison
    print(f"Random Forest:")
    print(f"  R² Score: {rf_results['r2']:.4f}")
    print(f"  MAE: {rf_results['mae']:,.0f} metric tons")
    print(f"  RMSE: {rf_results['rmse']:,.0f} metric tons")
    
    print(f"\nLinear Regression:")
    print(f"  R² Score: {lr_results['r2']:.4f}")
    print(f"  MAE: {lr_results['mae']:,.0f} metric tons")
    print(f"  RMSE: {lr_results['rmse']:,.0f} metric tons")
    
    # Select best model
    if rf_results['r2'] > lr_results['r2']:
        best_model = rf_predictor
        best_results = rf_results
        print(f"R² = {rf_results['r2']:.4f}")
        best_model.save_model('data/models/trained_model.pkl')
    else:
        best_model = lr_predictor
        best_results = lr_results
        print(f"R² = {lr_results['r2']:.4f}")
        best_model.save_model('data/models/trained_model.pkl')
    
    # Step 6: Test predictions
    test_cases = [
        ("TX", "Power Plants", 2023),
        ("CA", "Oil & Gas", 2022),
        ("NY", "Waste", 2021),
    ]
    
    for state, sector, year in test_cases:
        try:
            prediction = best_model.predict(state, sector, year)
            print(f"  {state} {sector} ({year}): {prediction:,.0f} metric tons CO₂e")
        except Exception as e:
            print(f"  {state} {sector} ({year}): Error - {e}")
    
    

if __name__ == "__main__":
    main()
