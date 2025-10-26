import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def load_and_clean_data(file_path="data/raw/epa_ghgrp_2021_2023_aggregate.csv"):
    """Load and lightly clean EPA GHGRP aggregate data."""
    # resolve relative path
    if not os.path.exists(file_path):
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "raw", "epa_ghgrp_2021_2023_aggregate.csv"
        )

    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")

    # drop rows with missing core fields
    initial_rows = len(df)
    df = df.dropna(subset=["total_ghg_emissions_tonnes", "state", "industry_sector"])
    print(
        f"Rows after removing missing values: {len(df)} (removed: {initial_rows - len(df)})"
    )

    # remove zero or non-positive emissions
    df = df[df["total_ghg_emissions_tonnes"] > 0]
    print(f"Rows after removing zero emissions: {len(df)}")

    # fill lat/lon by state median then overall median
    for geo_col in ("latitude", "longitude"):
        if geo_col in df.columns:
            df[geo_col] = df.groupby("state")[geo_col].transform(
                lambda s: s.fillna(s.median())
            )
            if df[geo_col].isna().any():
                df[geo_col] = df[geo_col].fillna(df[geo_col].median())

    # normalize sector names
    df["industry_sector_clean"] = df["industry_sector"].apply(clean_industry_sector)
    return df


def clean_industry_sector(sector):
    """Return a cleaned/standardized sector label."""
    if pd.isna(sector):
        return "Other"
    sector = str(sector).strip()
    if "," in sector:
        sector = sector.split(",")[0].strip()
    sector_mapping = {
        "Petroleum and Natural Gas Systems": "Oil & Gas",
        "Natural Gas and Natural Gas Liquids Suppliers": "Oil & Gas",
        "Pulp and Paper": "Pulp & Paper",
    }
    return sector_mapping.get(sector, sector)


def prepare_features(
    df,
    target_col="total_ghg_emissions_tonnes",
    add_interactions=True,
    add_aggregates=True,
    log_transform_numeric=False,
):
    """Prepare features for modeling.

    Parameters
    - df: DataFrame with raw data (should include cleaned industry_sector_clean)
    - target_col: name of the target column (used to compute aggregates)
    - add_interactions: create state x sector interaction feature
    - add_aggregates: add aggregated features (mean emissions by state/sector)
    - log_transform_numeric: apply log1p to strictly positive numeric feature columns
    """
    # Create a copy for feature engineering
    base_features = ["state", "industry_sector_clean", "reporting_year"]

    # Add geolocation features if available
    geo_features = [c for c in ("latitude", "longitude") if c in df.columns]

    features_df = df[base_features + geo_features].copy()

    # Interaction: state x sector
    if add_interactions:
        features_df["state_x_sector"] = (
            features_df["state"].astype(str)
            + "|"
            + features_df["industry_sector_clean"].astype(str)
        )

    # Aggregated features (facility-scale proxies) computed from available target values
    if add_aggregates and target_col in df.columns:
        # mean emissions by state
        state_means = df.groupby("state")[target_col].mean()
        features_df["state_mean_emissions"] = (
            features_df["state"].map(state_means).fillna(0.0)
        )

        # mean emissions by sector
        sector_means = df.groupby("industry_sector_clean")[target_col].mean()
        features_df["sector_mean_emissions"] = (
            features_df["industry_sector_clean"].map(sector_means).fillna(0.0)
        )

        # combined state-sector mean
        state_sector_means = df.groupby(["state", "industry_sector_clean"])[
            target_col
        ].mean()
        features_df["state_sector_mean_emissions"] = features_df.apply(
            lambda r: state_sector_means.get(
                (r["state"], r["industry_sector_clean"]), np.nan
            ),
            axis=1,
        ).fillna(0.0)

    # Optionally log-transform numeric features that are strictly positive
    if log_transform_numeric:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            col_vals = features_df[col]
            if (col_vals > 0).all():
                features_df[col] = np.log1p(col_vals)

    target = df[target_col].copy() if target_col in df.columns else None

    return features_df, target


def encode_categorical_features(X_train, X_test):
    """Encode categorical features for machine learning"""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Label encode categorical variables
    label_encoders = {}

    # include the interaction feature if present
    categorical_cols = ["state", "industry_sector_clean"]
    if "state_x_sector" in X_train.columns:
        categorical_cols.append("state_x_sector")

    for col in categorical_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))

        # Handle unseen categories in test set
        X_test_encoded[col] = X_test[col].map(
            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
        )

        label_encoders[col] = le

    return X_train_encoded, X_test_encoded, label_encoders


def get_top_emitters(df, by="total", top_n=10):
    """Get top emitters by different criteria"""
    if by == "total":
        return df.nlargest(top_n, "total_ghg_emissions_tonnes")
    elif by == "sector":
        return (
            df.groupby("industry_sector_clean")["total_ghg_emissions_tonnes"]
            .sum()
            .nlargest(top_n)
        )
    elif by == "state":
        return df.groupby("state")["total_ghg_emissions_tonnes"].sum().nlargest(top_n)


def calculate_outliers(df, column="total_ghg_emissions_tonnes", method="iqr"):
    """Calculate outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outliers, lower_bound, upper_bound
