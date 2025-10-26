import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.dirname(__file__))

from data_processing import load_and_clean_data, prepare_features
from visualizations import (plot_top_sectors, plot_state_map, plot_yearly_trends, 
                           plot_outliers, plot_model_performance, create_summary_dashboard)
from model import EmissionsPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="GHG Emissions Prediction Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for shadcn integration
st.markdown("""
<style>
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        line-height: 1.6;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: hsl(210 40% 98%);
        border-radius: 6px;
        color: hsl(222.2 84% 4.9%);
        border: 1px solid hsl(214.3 31.8% 91.4%);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: hsl(222.2 84% 4.9%);
        color: hsl(210 40% 98%);
    }
    
    h1, h2, h3 {
        font-weight: 600;
        color: hsl(222.2 84% 4.9%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    return load_and_clean_data()

@st.cache_resource
def load_or_train_model(df):
    """Load existing model or train a new one"""
    try:
        # Try to load existing model
        predictor = EmissionsPredictor()
        predictor.load_model()
        return predictor, None
    except Exception as e:
        # Train new model
        predictor = EmissionsPredictor()
        X, y = prepare_features(df)
        
        with st.spinner('Training Random Forest model...'):
            predictor.fit(X, y)
            predictor.save_model()
        
        return predictor, X

def main():
    # Main header using shadcn card
    ui.card(
        content="""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; color: hsl(222.2 84% 4.9%);">
                GHG Emissions Prediction Portal
            </h1>
            <p style="font-size: 1.1rem; color: hsl(215.4 16.3% 46.9%); margin: 0;">
                Advanced analytics and prediction system for greenhouse gas emissions
            </p>
        </div>
        """,
        key="main_header"
    )
    
    # Load data
    try:
        df = load_data()
        predictor, X_train = load_or_train_model(df)
        
        # Alert for successful data loading
        ui.alert(
            title="System Status",
            description=f"Successfully loaded {len(df):,} records from EPA GHGRP database",
            alert_type="default",
            key="data_status"
        )
        
    except Exception as e:
        ui.alert(
            title="Error Loading Data",
            description=f"Failed to load data: {str(e)}",
            alert_type="destructive",
            key="error_alert"
        )
        st.stop()
    
    # Create tabs using shadcn-ui
    selected_tab = ui.tabs(
        options=["Dashboard", "Sector Analysis", "Geographic Analysis", "Time Series", "Predictions", "Model Performance"],
        default_value="Dashboard",
        key="main_tabs"
    )
    
    if selected_tab == "Dashboard":
        st.header("Executive Dashboard")
        
        # Key metrics using shadcn metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ui.metric_card(
                title="Total Records",
                content=f"{len(df):,}",
                description="EPA GHGRP entries",
                key="metric_records"
            )
        
        with col2:
            ui.metric_card(
                title="Total Emissions",
                content=f"{df['total_reported_emissions'].sum():,.0f}",
                description="MT CO2e",
                key="metric_emissions"
            )
        
        with col3:
            ui.metric_card(
                title="Active Facilities",
                content=f"{df['facility_name'].nunique():,}",
                description="Unique facilities",
                key="metric_facilities"
            )
        
        with col4:
            ui.metric_card(
                title="Industry Sectors",
                content=f"{df['industry_type'].nunique():,}",
                description="Different sectors",
                key="metric_sectors"
            )
        
        # Dashboard visualization
        st.subheader("Overview")
        fig_dashboard = create_summary_dashboard(df)
        st.plotly_chart(fig_dashboard, use_container_width=True)
    
    elif selected_tab == "Sector Analysis":
        st.header("Industry Sector Analysis")
        
        # Sector selection using shadcn select
        sectors = sorted(df['industry_type'].unique())
        selected_sector = ui.select(
            options=sectors,
            default_value=sectors[0] if sectors else None,
            placeholder="Select an industry sector",
            key="sector_select"
        )
        
        if selected_sector:
            sector_data = df[df['industry_type'] == selected_sector]
            
            # Sector info card
            ui.card(
                content=f"""
                <h3>{selected_sector}</h3>
                <p><strong>Facilities:</strong> {len(sector_data):,}</p>
                <p><strong>Total Emissions:</strong> {sector_data['total_reported_emissions'].sum():,.0f} MT CO2e</p>
                <p><strong>Average per Facility:</strong> {sector_data['total_reported_emissions'].mean():,.0f} MT CO2e</p>
                """,
                key="sector_info"
            )
        
        # Top sectors visualization
        st.subheader("Top Emitting Sectors")
        fig_sectors = plot_top_sectors(df)
        st.plotly_chart(fig_sectors, use_container_width=True)
    
    elif selected_tab == "Geographic Analysis":
        st.header("Geographic Distribution")
        
        # State selection
        states = sorted(df['state'].unique())
        selected_state = ui.select(
            options=states,
            default_value=None,
            placeholder="Select a state (optional)",
            key="state_select"
        )
        
        if selected_state:
            state_data = df[df['state'] == selected_state]
            
            # State info card
            ui.card(
                content=f"""
                <h3>{selected_state}</h3>
                <p><strong>Facilities:</strong> {len(state_data):,}</p>
                <p><strong>Total Emissions:</strong> {state_data['total_reported_emissions'].sum():,.0f} MT CO2e</p>
                <p><strong>Top Sector:</strong> {state_data.groupby('industry_type')['total_reported_emissions'].sum().idxmax()}</p>
                """,
                key="state_info"
            )
        
        # Geographic map
        st.subheader("State-wise Emission Distribution")
        fig_map = plot_state_map(df)
        st.plotly_chart(fig_map, use_container_width=True)
    
    elif selected_tab == "Time Series":
        st.header("Temporal Analysis")
        
        # Year selection using shadcn slider
        min_year = int(df['reporting_year'].min())
        max_year = int(df['reporting_year'].max())
        
        selected_years = ui.slider(
            min_value=min_year,
            max_value=max_year,
            default_value=[min_year, max_year],
            step=1,
            key="year_slider"
        )
        
        # Filter data by selected years
        if selected_years:
            filtered_df = df[
                (df['reporting_year'] >= selected_years[0]) & 
                (df['reporting_year'] <= selected_years[1])
            ]
            
            # Time series info
            ui.card(
                content=f"""
                <h3>Time Period: {selected_years[0]} - {selected_years[1]}</h3>
                <p><strong>Records:</strong> {len(filtered_df):,}</p>
                <p><strong>Total Emissions:</strong> {filtered_df['total_reported_emissions'].sum():,.0f} MT CO2e</p>
                <p><strong>Trend:</strong> {'Increasing' if filtered_df.groupby('reporting_year')['total_reported_emissions'].sum().iloc[-1] > filtered_df.groupby('reporting_year')['total_reported_emissions'].sum().iloc[0] else 'Decreasing'}</p>
                """,
                key="time_info"
            )
            
            # Yearly trends visualization
            st.subheader("Emission Trends Over Time")
            fig_trends = plot_yearly_trends(filtered_df)
            st.plotly_chart(fig_trends, use_container_width=True)
    
    elif selected_tab == "Predictions":
        st.header("Emissions Prediction")
        
        # Prediction form using shadcn components
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Facility Information")
                
                # Industry type selection
                industry_options = sorted(df['industry_type'].unique())
                industry_type = st.selectbox("Industry Type", industry_options)
                
                # State selection
                state_options = sorted(df['state'].unique())
                state = st.selectbox("State", state_options)
                
                # Facility size (based on historical data)
                facility_size = st.selectbox(
                    "Facility Size",
                    ["Small (< 25k MT CO2e)", "Medium (25k-100k MT CO2e)", "Large (> 100k MT CO2e)"]
                )
            
            with col2:
                st.subheader("Operational Parameters")
                
                # Reporting year
                reporting_year = st.number_input(
                    "Reporting Year",
                    min_value=2020,
                    max_value=2030,
                    value=2024
                )
                
                # Additional features based on your model
                if 'parent_company' in df.columns:
                    company_options = sorted(df['parent_company'].unique()[:100])  # Top 100
                    parent_company = st.selectbox("Parent Company", company_options)
            
            submitted = st.form_submit_button("Predict Emissions")
            
            if submitted:
                try:
                    # Create prediction input
                    facility_data = {
                        'industry_type': industry_type,
                        'state': state,
                        'reporting_year': reporting_year,
                        'facility_size': facility_size
                    }
                    
                    # Make prediction (you'll need to adapt this based on your model's features)
                    # This is a simplified example
                    sample_features = df[
                        (df['industry_type'] == industry_type) & 
                        (df['state'] == state)
                    ].iloc[0:1]  # Get a sample for feature structure
                    
                    if len(sample_features) > 0:
                        X_sample, _ = prepare_features(sample_features)
                        prediction = predictor.predict(X_sample)[0]
                        
                        # Display prediction result using shadcn card
                        ui.card(
                            content=f"""
                            <div style="text-align: center; padding: 2rem;">
                                <h2 style="color: hsl(142.1 76.2% 36.3%); margin-bottom: 1rem;">
                                    Predicted Emissions
                                </h2>
                                <div style="font-size: 3rem; font-weight: bold; color: hsl(222.2 84% 4.9%); margin: 1rem 0;">
                                    {prediction:,.0f} MT CO2e
                                </div>
                                <p style="color: hsl(215.4 16.3% 46.9%);">
                                    Estimated annual greenhouse gas emissions
                                </p>
                            </div>
                            """,
                            key="prediction_result"
                        )
                        
                        # Benchmark comparison
                        sector_avg = df[df['industry_type'] == industry_type]['total_reported_emissions'].mean()
                        state_avg = df[df['state'] == state]['total_reported_emissions'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            ui.metric_card(
                                title="vs Sector Average",
                                content=f"{((prediction - sector_avg) / sector_avg * 100):+.1f}%",
                                description=f"Sector avg: {sector_avg:,.0f} MT CO2e",
                                key="sector_comparison"
                            )
                        
                        with col2:
                            ui.metric_card(
                                title="vs State Average",
                                content=f"{((prediction - state_avg) / state_avg * 100):+.1f}%",
                                description=f"State avg: {state_avg:,.0f} MT CO2e",
                                key="state_comparison"
                            )
                    
                except Exception as e:
                    ui.alert(
                        title="Prediction Error",
                        description=f"Unable to generate prediction: {str(e)}",
                        alert_type="destructive",
                        key="prediction_error"
                    )
    
    elif selected_tab == "Model Performance":
        st.header("Model Performance Metrics")
        
        if X_train is not None:
            # Model info card
            ui.card(
                content=f"""
                <h3>Random Forest Model</h3>
                <p><strong>Training Records:</strong> {len(X_train):,}</p>
                <p><strong>Features:</strong> {X_train.shape[1]}</p>
                <p><strong>Algorithm:</strong> Random Forest Regressor</p>
                <p><strong>Status:</strong> Ready for predictions</p>
                """,
                key="model_info"
            )
            
            # Performance visualization
            st.subheader("Model Performance Analysis")
            try:
                fig_performance = plot_model_performance(predictor, X_train, df['total_reported_emissions'])
                st.plotly_chart(fig_performance, use_container_width=True)
            except Exception as e:
                ui.alert(
                    title="Performance Metrics Unavailable",
                    description="Unable to generate performance metrics. Model may need retraining.",
                    alert_type="default",
                    key="performance_error"
                )
        else:
            ui.alert(
                title="Model Training Required",
                description="Please run the model training process to view performance metrics.",
                alert_type="default",
                key="training_required"
            )
    
    # Footer
    st.markdown("---")
    ui.card(
        content="""
        <div style="text-align: center; padding: 1rem; color: hsl(215.4 16.3% 46.9%);">
            <p style="margin: 0;">GHG Emissions Prediction Portal | Built with Streamlit & shadcn/ui</p>
            <p style="margin: 0; font-size: 0.9rem;">Data source: EPA Greenhouse Gas Reporting Program (GHGRP)</p>
        </div>
        """,
        key="footer"
    )

if __name__ == "__main__":
    main()
