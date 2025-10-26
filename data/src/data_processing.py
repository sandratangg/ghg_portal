import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean_data(file_path='data/raw/epa_ghgrp_2021_2023_aggregate.csv'):
    """Load and clean EPA GHGRP aggregate data"""
    # Handle relative path from different locations
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'raw', 'epa_ghgrp_2021_2023_aggregate.csv')
    
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna(subset=['total_ghg_emissions_tonnes', 'state', 'industry_sector'])
    print(f"Rows after removing missing values: {len(df)} (removed: {initial_rows - len(df)})")
    
    # Remove zero emissions (these might be data quality issues)
    df = df[df['total_ghg_emissions_tonnes'] > 0]
    print(f"Rows after removing zero emissions: {len(df)}")
    
    # Clean up industry sectors (handle combined categories)
    df['industry_sector_clean'] = df['industry_sector'].apply(clean_industry_sector)
    
    return df

def clean_industry_sector(sector):
    """Clean and standardize industry sector names"""
    if pd.isna(sector):
        return 'Other'
    
    sector = str(sector).strip()
    
    # Handle combined sectors - take the first one
    if ',' in sector:
        sector = sector.split(',')[0].strip()
    
    # Standardize common variations
    sector_mapping = {
        'Petroleum and Natural Gas Systems': 'Oil & Gas',
        'Natural Gas and Natural Gas Liquids Suppliers': 'Oil & Gas',
        'Power Plants': 'Power Plants',
        'Waste': 'Waste',
        'Chemicals': 'Chemicals',
        'Metals': 'Metals',
        'Minerals': 'Minerals',
        'Pulp and Paper': 'Pulp & Paper',
        'Other': 'Other'
    }
    
    return sector_mapping.get(sector, sector)

def prepare_features(df):
    """Prepare features for modeling"""
    # Create a copy for feature engineering
    features_df = df[['state', 'industry_sector_clean', 'reporting_year']].copy()
    target = df['total_ghg_emissions_tonnes'].copy()
    
    return features_df, target

def encode_categorical_features(X_train, X_test):
    """Encode categorical features for machine learning"""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # Label encode categorical variables
    label_encoders = {}
    
    categorical_cols = ['state', 'industry_sector_clean']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
        
        # Handle unseen categories in test set
        X_test_encoded[col] = X_test[col].map(lambda x: le.transform([str(x)])[0] 
                                             if str(x) in le.classes_ else -1)
        
        label_encoders[col] = le
    
    return X_train_encoded, X_test_encoded, label_encoders

def get_top_emitters(df, by='total', top_n=10):
    """Get top emitters by different criteria"""
    if by == 'total':
        return df.nlargest(top_n, 'total_ghg_emissions_tonnes')
    elif by == 'sector':
        return df.groupby('industry_sector_clean')['total_ghg_emissions_tonnes'].sum().nlargest(top_n)
    elif by == 'state': 
        return df.groupby('state')['total_ghg_emissions_tonnes'].sum().nlargest(top_n)
    
def calculate_outliers(df, column='total_ghg_emissions_tonnes', method='iqr'):
    """Calculate outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound