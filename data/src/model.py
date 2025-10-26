import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

class EmissionsPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)
        self.label_encoders = {}
        self.scaler = None

    def preprocess(self, X: pd.DataFrame, fit=False):
        X_encoded = X.copy()

        # Encode categorical features
        for col in X_encoded.select_dtypes(include=['object']).columns:
            if fit or col not in self.label_encoders:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X_encoded[col] = le.transform(X_encoded[col])

        # Scale numeric features
        if fit:
            self.scaler = StandardScaler()
            X_encoded = pd.DataFrame(self.scaler.fit_transform(X_encoded), columns=X_encoded.columns)
        else:
            X_encoded = pd.DataFrame(self.scaler.transform(X_encoded), columns=X_encoded.columns)

        return X_encoded

    def fit(self, X, y):
        X_encoded = self.preprocess(X, fit=True)
        self.model.fit(X_encoded, y)
        return self

    def predict(self, X):
        X_encoded = self.preprocess(X, fit=False)
        return self.model.predict(X_encoded)

    def evaluate(self, y_true, y_pred):
        metrics = {
            "R2 Score": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }
        return metrics

    def save_model(self, path="data/models/trained_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "encoders": self.label_encoders,
            "scaler": self.scaler
        }, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="data/models/trained_model.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["encoders"]
        self.scaler = data["scaler"]
        print(f"📦 Loaded model from {path}")
