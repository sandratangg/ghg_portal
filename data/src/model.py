from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import pandas as pd
import numpy as np
import os

class EmissionsPredictor:
    def __init__(self, model_type='random_forest', tune_hyperparameters=False):
        """Initialize the predictor with specified model type"""
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        
        self.model_type = model_type
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.tune_hyperparameters = tune_hyperparameters

    def train(self, X, y):
        """
        Train the model with comprehensive evaluation and optional hyperparameter tuning.
        """
        print(f"Training {self.model_type} model...")
        print(f"Training data shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Encode categorical features
        X_encoded = self._encode_features(X, fit=True)

        # Split data (80% train, 20% test as specified)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=X['industry_sector_clean']
        )

        print(f"Train set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Linear regression branch with scaling
        if self.model_type == 'linear':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        # Random forest with optional hyperparameter tuning
        elif self.model_type == 'random_forest' and self.tune_hyperparameters:
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print("Best parameters from GridSearchCV:", grid_search.best_params_)
            y_pred = self.model.predict(X_test)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

        # Model evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Cross-validation
        if self.model_type == 'linear':
            cv_scores = cross_val_score(self.model, self.scaler.transform(X_encoded), y, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(self.model, X_encoded, y, cv=5, scoring='r2')

        # Feature importance (for Random Forest)
        feature_importance = None
        if self.model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': [f'encoded_{col}' if col in ['state', 'industry_sector_clean'] else col
                            for col in X_encoded.columns],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        results = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': feature_importance
        }

        print(f"\n=== Model Performance ===")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:,.2f} metric tons")
        print(f"RMSE: {rmse:,.2f} metric tons")
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if feature_importance is not None:
            print(f"\n=== Top 5 Most Important Features ===")
            print(feature_importance.head())

        return results

    def _encode_features(self, X, fit=False):
        """Encode categorical features"""
        X_encoded = X.copy()
        categorical_cols = ['state', 'industry_sector_clean']
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories
                X_encoded[col] = X[col].map(
                    lambda x: self.encoders[col].transform([str(x)])[0] 
                    if str(x) in self.encoders[col].classes_ else -1
                )
        return X_encoded

    def predict(self, state, sector, year):
        """Predict emissions for a given facility"""
        X = pd.DataFrame({
            'state': [state],
            'industry_sector_clean': [sector],
            'reporting_year': [year]
        })
        # Encode categorical features
        X_encoded = self._encode_features(X, fit=False)
        # Handle unknown categories (set to -1)
        X_encoded = X_encoded.fillna(-1)
        if self.model_type == 'linear':
            X_scaled = self.scaler.transform(X_encoded)
            prediction = self.model.predict(X_scaled)[0]
        else:
            prediction = self.model.predict(X_encoded)[0]
        return max(0, prediction)

    def get_feature_importance(self):
        """Get feature importance (Random Forest only)"""
        if self.model_type != 'random_forest':
            return None
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def save_model(self, path='../models/trained_model.pkl'):
        """Save model and encoders"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path='../models/trained_model.pkl'):
        """Load trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        predictor = cls(model_type=data.get('model_type', 'random_forest'))
        predictor.model = data['model']
        predictor.encoders = data['encoders']
        predictor.scaler = data.get('scaler', StandardScaler())
        predictor.feature_names = data.get('feature_names', [])
        return predictor

def train_and_save_model(data_path='../raw/epa_ghgrp_2021_2023_aggregate.csv'):
    """Complete pipeline to train and save the model"""
    from data_processing import load_and_clean_data, prepare_features
    
    # Load and prepare data
    df = load_and_clean_data(data_path)
    X, y = prepare_features(df)
    
    # Train Random Forest model
    print("Training Random Forest Model...")
    rf_predictor = EmissionsPredictor(model_type='random_forest')
    rf_results = rf_predictor.train(X, y)
    rf_predictor.save_model('../models/random_forest_model.pkl')
    
    # Train Linear Regression baseline
    print("\nTraining Linear Regression Baseline...")
    lr_predictor = EmissionsPredictor(model_type='linear')
    lr_results = lr_predictor.train(X, y)
    lr_predictor.save_model('../models/linear_model.pkl')
    
    # Compare models
    print(f"\n=== Model Comparison ===")
    print(f"Random Forest - R²: {rf_results['r2']:.4f}, MAE: {rf_results['mae']:,.2f}")
    print(f"Linear Regression - R²: {lr_results['r2']:.4f}, MAE: {lr_results['mae']:,.2f}")
    
    # Return the better model
    if rf_results['r2'] > lr_results['r2']:
        print("Random Forest performs better - using as primary model")
        return rf_predictor, rf_results
    else:
        print("Linear Regression performs better - using as primary model")
        return lr_predictor, lr_results
