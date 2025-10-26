import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add the data/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "data", "src"))
from model import EmissionsPredictor
from data_processing import load_and_clean_data, prepare_features


# Use the data_processing helpers to load and prepare features
def load_data_and_prepare(
    add_interactions=True, add_aggregates=True, log_features=False
):
    df = load_and_clean_data()
    X, y = prepare_features(
        df,
        add_interactions=add_interactions,
        add_aggregates=add_aggregates,
        log_transform_numeric=log_features,
    )
    return X, y, df


def main():
    # parse CLI flags
    args = sys.argv[1:]
    grid_search_flag = "--grid" in args
    randomized_flag = "--randomized" in args or "--random" in args
    n_iter = 50
    cv = 5
    log_target = "--logtarget" in args

    # feature-engineering flags
    add_interactions = "--no_interactions" not in args
    add_aggregates = "--no_aggregates" not in args
    log_features = "--log_features" in args

    # per-sector quick models
    per_sector_flag = "--per_sector" in args

    # allow override via flags like --n_iter=100 and --cv=10
    for a in args:
        if a.startswith("--n_iter="):
            try:
                n_iter = int(a.split("=", 1)[1])
            except:
                pass
        if a.startswith("--cv="):
            try:
                cv = int(a.split("=", 1)[1])
            except:
                pass

    # Always prepare base features without aggregates to avoid accidental leakage.
    X, y, df = load_data_and_prepare(
        add_interactions=add_interactions,
        add_aggregates=False,
        log_features=log_features,
    )

    # If aggregates requested, compute them using a holdout of the training partition
    # so aggregated features are derived only from training targets (avoid leakage).
    if add_aggregates:
        # initial split to compute aggregates (80% train, 20% holdout)
        X_tmp_train, X_tmp_hold, y_tmp_train, y_tmp_hold = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build a small DataFrame to compute group means
        tmp_df = pd.DataFrame(
            {
                "state": X_tmp_train["state"].values,
                "industry_sector_clean": X_tmp_train["industry_sector_clean"].values,
                "target": y_tmp_train.values,
            }
        )

        state_means = tmp_df.groupby("state")["target"].mean()
        sector_means = tmp_df.groupby("industry_sector_clean")["target"].mean()
        state_sector_means = tmp_df.groupby(["state", "industry_sector_clean"])[
            "target"
        ].mean()

        # Map aggregates back onto the full feature set
        X = X.copy()
        X["state_mean_emissions"] = X["state"].map(state_means).fillna(0.0)
        X["sector_mean_emissions"] = (
            X["industry_sector_clean"].map(sector_means).fillna(0.0)
        )
        X["state_sector_mean_emissions"] = X.apply(
            lambda r: state_sector_means.get(
                (r["state"], r["industry_sector_clean"]), np.nan
            ),
            axis=1,
        ).fillna(0.0)

    # Train-test split
    predictor = EmissionsPredictor()

    # predictor already created above

    if grid_search_flag:
        print("\nTraining model with GridSearchCV (this may take several minutes)...")
        results = predictor.train(X, y, grid_search=True)
    elif randomized_flag:
        print(f"\nTraining model with RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")

        # Prepare train/test split and encoding
        if log_target:
            y_used = np.log1p(y)
        else:
            y_used = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_used, test_size=0.2, random_state=42
        )

        # Fit preprocessing on training data
        X_train_encoded = predictor.preprocess(X_train, fit=True)
        X_test_encoded = predictor.preprocess(X_test, fit=False)

        param_dist = {
            "n_estimators": randint(100, 1000),
            "max_depth": [None, 10, 20, 40, 80],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
        }

        rnd = RandomizedSearchCV(
            predictor.model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=2,
        )
        rnd.fit(X_train_encoded, y_train)
        predictor.model = rnd.best_estimator_

        # Predict and compute metrics
        y_pred = predictor.model.predict(X_test_encoded)

        if log_target:
            # back-transform
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            r2 = float(r2_score(y_test_orig, y_pred_orig))
            mae = float(mean_absolute_error(y_test_orig, y_pred_orig))
            rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
        else:
            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        results = {"r2": r2, "mae": mae, "rmse": rmse}

        print(f"\nBest Parameters: {rnd.best_params_}")
        print(f"Best CV Score (approx): {rnd.best_score_:.4f}")
    else:
        print("\nTraining model (quick mode: grid_search=False)...")
        results = predictor.train(X, y, grid_search=False)

    # Optionally train per-sector models (quick runs, skip tiny sectors)
    if per_sector_flag:
        sectors = df["industry_sector_clean"].unique()
        print(
            "\nTraining per-sector models (quick mode). This will train one model per sector with >= 50 rows."
        )
        sector_metrics = {}
        for s in sectors:
            subset = df[df["industry_sector_clean"] == s]
            if len(subset) < 50:
                continue
            Xs, ys = prepare_features(
                subset,
                add_interactions=add_interactions,
                add_aggregates=False,
                log_transform_numeric=log_features,
            )
            sec_pred = EmissionsPredictor()
            sec_res = sec_pred.train(Xs, ys, grid_search=False)
            sector_metrics[s] = {"r2": sec_res["r2"], "mae": sec_res["mae"]}
            # save quick sector model
            safe_name = s.replace(" ", "_").replace("/", "_")[:40]
            sec_pred.save_model(f"data/models/trained_model_sector_{safe_name}.pkl")

        print("\nPer-sector training complete. Summary:")
        for s, m in sector_metrics.items():
            print(f" - {s}: R2={m['r2']:.4f}, MAE={m['mae']:.0f}")

    print("\nFinal Test Metrics:")
    print(f" - R2: {results['r2']:.4f}")
    print(f" - MAE: {results['mae']:.4f}")
    print(f" - RMSE: {results['rmse']:.4f}")

    # Save model
    predictor.save_model("data/models/trained_model.pkl")
    print("\nTraining complete! Model saved successfully.")


if __name__ == "__main__":
    main()
