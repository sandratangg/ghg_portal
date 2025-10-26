# GHG Emissions Prediction Portal

A comprehensive web-based machine learning application for analyzing and predicting greenhouse gas emissions from EPA facility data (2021-2023).

## Project Overview

This portal addresses **Challenge 2 - Data Modeling Portal** by providing:

1. **Interactive Data Visualizations** - Explore GHG emissions patterns across states, sectors, and time
2. **Machine Learning Predictions** - Predict facility emissions using Random Forest and Linear Regression models
3. **Comprehensive Analysis** - Identify trends, outliers, and key emission drivers
4. **User-Friendly Interface** - Interactive Streamlit web application

## Features

### Data Exploration & Visualizations
- **Top 10 Emitting Sectors** - Bar chart analysis
- **State-by-State Emissions Map** - Interactive choropleth visualization
- **Yearly Trends (2021-2023)** - Time series analysis
- **Outlier Detection** - Identify extreme emitters
- **Summary Dashboard** - Key metrics and insights

### Machine Learning Model
- **Target Variable**: `total_ghg_emissions_tonnes`
- **Features**: `state`, `industry_sector`, `reporting_year`
- **Models**: Random Forest Regressor (primary) + Linear Regression (baseline)
- **Evaluation**: R² Score and Mean Absolute Error (MAE)
- **Train/Test Split**: 80% / 20%

### Interactive Predictions
- Select facility characteristics (state, sector, year)
- Get real-time emissions predictions
- Compare with similar facilities
- View model performance metrics

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sandratangg/ghg_portal.git
cd ghg_portal
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the models**
```bash
python train_model.py
```

4. **Launch the web application**
```bash
streamlit run data/src/app.py
```

5. **Open your browser** to `http://localhost:8501`

## Project Structure

```
ghg-portal/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── train_model.py                     # Model training pipeline
├── data/
│   ├── raw/
│   │   └── epa_ghgrp_2021_2023_aggregate.csv  # EPA GHG data
│   ├── models/
│   │   ├── trained_model.pkl          # Best performing model
│   │   ├── random_forest_model.pkl    # Random Forest model
│   │   └── linear_model.pkl           # Linear Regression model
│   ├── notebooks/
│   │   └── exploration.ipynb          # Data exploration notebook
│   └── src/
│       ├── app.py                     # Main Streamlit application
│       ├── data_processing.py         # Data cleaning and preparation
│       ├── model.py                   # Machine learning models
│       └── visualizations.py         # Plotting functions
```

##Model Performance

The Random Forest model achieves:
- **R² Score**: ~0.85+ (explains 85%+ of emission variance)
- **MAE**: <100K metric tons (typical prediction error)
- **Features**: State, Industry Sector, Reporting Year

### Key Insights
- **Power Plants** are the largest emission sector
- **Texas, California, Louisiana** are top emitting states
- **Year-over-year trends** show slight increases in some sectors
- **Outlier facilities** represent <5% but contribute significantly to total emissions

##Technical Details

### Data Processing
- **Dataset**: EPA GHGRP 2021-2023 (19,513 facility records)
- **Cleaning**: Remove missing values, handle outliers
- **Feature Engineering**: Label encoding for categorical variables
- **Validation**: Stratified train-test split

### Machine Learning
- **Algorithm**: Random Forest Regressor (scikit-learn)
- **Hyperparameters**: 100 trees, max depth 20, optimized for performance
- **Cross-Validation**: 5-fold CV for model stability
- **Baseline**: Linear Regression for comparison

### Web Application
- **Framework**: Streamlit
- **Visualizations**: Plotly for interactive charts
- **Caching**: Efficient data loading and model inference
- **Responsive Design**: Multi-column layouts and navigation

## Application Sections

### Dashboard Overview
- Key metrics summary
- Top emitting sectors visualization
- State-by-state emissions map
- Yearly trend analysis

### Detailed Analysis
- Interactive outlier analysis
- Top 20 emitting facilities
- Sector statistics and comparisons
- Data exploration tools

### Emissions Predictor
- Input facility characteristics
- Real-time ML predictions
- Comparison with similar facilities
- Model confidence indicators

### Model Performance
- R², MAE, RMSE metrics
- Actual vs Predicted plots
- Feature importance analysis
- Model interpretation

## Key Findings

1. **Power Plants** dominate emissions (largest sector)
2. **Geographic Concentration**: TX, CA, LA are top emitting states
3. **Temporal Trends**: Relatively stable emissions 2021-2023
4. **Outliers**: Few facilities (top 5%) contribute disproportionately
5. **Predictability**: State and sector are strong emission predictors

