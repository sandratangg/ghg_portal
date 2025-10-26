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

        # Encode categorical features. Store encoders and handle unseen values.
        for col in X_encoded.select_dtypes(include=['object']).columns:
            if fit or col not in self.label_encoders:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                # store encoder and set of known classes
                self.label_encoders[col] = {'le': le, 'classes': set(le.classes_)}
            else:
                info = self.label_encoders[col]
                le = info['le']
                classes = info.get('classes', set())
                # map unseen values to -1
                X_encoded[col] = (
                    X_encoded[col]
                    .astype(str)
                    .map(lambda v: le.transform([v])[0] if v in classes else -1)
                )

        # Ensure column ordering and presence when transforming
        if not fit and hasattr(self, "_feature_names"):
            # add any missing features with zeros and ensure column order
            for c in self._feature_names:
                if c not in X_encoded.columns:
                    X_encoded[c] = 0.0
            # keep only expected columns in the correct order
            X_encoded = X_encoded.reindex(columns=self._feature_names)

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
        # Accept DataFrame, list/tuple (3 or 5 items), or dict
        if isinstance(X, pd.DataFrame):
            X_encoded = self.preprocess(X, fit=False)
            preds = self.model.predict(X_encoded)
            return preds
        elif isinstance(X, (list, tuple)):
            # Support (state, sector, year) or (state, sector, year, latitude, longitude)
            if len(X) == 3:
                state, sector, year = X
                row = {
                    "state": state,
                    "industry_sector_clean": sector,
                    "reporting_year": year,
                }
            elif len(X) == 5:
                state, sector, year, latitude, longitude = X
                row = {
                    "state": state,
                    "industry_sector_clean": sector,
                    "reporting_year": year,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            else:
                raise ValueError("Unsupported input tuple length for predict(). Use 3 or 5 elements.")

            df = pd.DataFrame([row])
            X_encoded = self.preprocess(df, fit=False)
            return float(self.model.predict(X_encoded)[0])
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
            X_encoded = self.preprocess(df, fit=False)
            return float(self.model.predict(X_encoded)[0])
        else:
            raise ValueError("Unsupported input for predict(). Pass a DataFrame, dict, or tuple/list.")

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
            "scaler": self.scaler,
            "feature_names": getattr(self, "_feature_names", None),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path="data/models/trained_model.pkl"):
        data = joblib.load(path)
        self.model = data.get("model", self.model)
        self.label_encoders = data.get("encoders", {})
        self.scaler = data.get("scaler", None)
        if data.get("feature_names") is not None:
            self._feature_names = data.get("feature_names")
        print(f"Loaded model from {path}")
    
    def get_feature_importance(self):
        """Get feature importance for RandomForest model"""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names from encoders
            feature_names = []
            for col in self.label_encoders.keys():
                feature_names.append(col)
            
            # Add any remaining numeric features
            if hasattr(self, '_feature_names'):
                feature_names = self._feature_names
            else:
                # Default feature names based on typical structure
                feature_names = ['state', 'industry_sector_clean', 'reporting_year']
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(self.model.feature_importances_)],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None

    def train(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42,
        cv=3,
        param_grid=None,
        grid_search=True,
    ):
        """Train the predictor. Returns a results dict with metrics and feature importance."""
        from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Fit encoders/scaler on training data
        X_train_encoded = self.preprocess(X_train, fit=True)
        X_test_encoded = self.preprocess(X_test, fit=False)

        self._feature_names = list(X_train_encoded.columns)

        cv_mean = None
        if grid_search:
            grid = GridSearchCV(self.model, param_grid, cv=cv, scoring="r2", n_jobs=-1)
            grid.fit(X_train_encoded, y_train)
            self.model = grid.best_estimator_
            cv_scores = cross_val_score(self.model, X_train_encoded, y_train, cv=cv, scoring="r2")
            cv_mean = float(np.mean(cv_scores))
        else:
            self.model.fit(X_train_encoded, y_train)

        # Predictions and metrics
        y_pred = self.model.predict(X_test_encoded)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "cv_mean": cv_mean,
        }

        # Feature importance
        feature_importance = None
        if hasattr(self.model, "feature_importances_"):
            try:
                feat_names = X_train_encoded.columns.tolist()
                feature_importance = pd.DataFrame(
                    {"feature": feat_names, "importance": self.model.feature_importances_}
                ).sort_values("importance", ascending=False)
            except Exception:
                feature_importance = None

        results = {
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "cv_mean": metrics["cv_mean"],
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
            "feature_importance": feature_importance,
        }

        return results
