import streamlit as st
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
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .section-header {
        color: #2E8B57;
        border-bottom: 2px solid #2E8B57;
        padding-bottom: 10px;
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
        st.success("Loaded pre-trained model")
        return predictor, None
    except Exception as e:
        # Train new model
        st.info("Training new model...")
        predictor = EmissionsPredictor()
        X, y = prepare_features(df)
        
        with st.spinner('Training Random Forest model...'):
            predictor.fit(X, y)
            y_pred = predictor.predict(X)
            results = predictor.evaluate(y, y_pred)
        
        predictor.save_model()
        st.success("Model trained successfully!")
        return predictor, results

def main():
    # Header
    st.markdown('<h1 class="main-header">GHG Emissions Prediction Portal</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **GHG Emissions Prediction Portal**! This interactive dashboard analyzes EPA greenhouse gas 
    reporting data from 2021-2023 and provides machine learning predictions for facility emissions.
    """)
    
    # Load data
    with st.spinner('Loading EPA GHG data...'):
        df = load_data()
    
    # Load/train model
    predictor, training_results = load_or_train_model(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard Overview", "Detailed Analysis", "Emissions Predictor", "Model Performance"]
    )
    
    if page == "Dashboard Overview":
        show_dashboard_overview(df)
    elif page == "Detailed Analysis":
        show_detailed_analysis(df)
    elif page == "Emissions Predictor":
        show_emissions_predictor(df, predictor)
    elif page == "Model Performance":
        show_model_performance(training_results, predictor)

def show_dashboard_overview(df):
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', 
                unsafe_allow_html=True)
    
    # Key metrics
    metrics = create_summary_dashboard(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Emissions", metrics['Total Emissions'])
    with col2:
        st.metric("Avg per Facility", metrics['Average per Facility'])
    with col3:
        st.metric("Total Facilities", metrics['Total Facilities'])
    with col4:
        st.metric("States Covered", metrics['States Covered'])
    with col5:
        st.metric("Industry Sectors", metrics['Industry Sectors'])
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Emitting Sectors")
        fig_sectors = plot_top_sectors(df, top_n=10)
        st.plotly_chart(fig_sectors, use_container_width=True)
    
    with col2:
        st.subheader("Emissions by State")
        fig_map = plot_state_map(df)
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Yearly trends
    st.subheader("Emissions Trends (2021-2023)")
    fig_trends = plot_yearly_trends(df)
    st.plotly_chart(fig_trends, use_container_width=True)

def show_detailed_analysis(df):
    st.markdown('<h2 class="section-header">Detailed Analysis</h2>', 
                unsafe_allow_html=True)
    
    # Outlier analysis
    st.subheader("Outlier Analysis")
    
    threshold = st.slider("Outlier Threshold (Percentile)", min_value=90, max_value=99, value=95)
    fig_outliers = plot_outliers(df, threshold_percentile=threshold)
    st.plotly_chart(fig_outliers, use_container_width=True)
    
    # Data exploration options
    st.subheader("Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top facilities
        st.subheader("Top 20 Emitting Facilities")
        top_facilities = df.nlargest(20, 'total_ghg_emissions_tonnes')[
            ['facility_name', 'state', 'industry_sector_clean', 'total_ghg_emissions_tonnes', 'reporting_year']
        ]
        top_facilities['emissions_millions'] = top_facilities['total_ghg_emissions_tonnes'] / 1_000_000
        st.dataframe(
            top_facilities[['facility_name', 'state', 'industry_sector_clean', 'emissions_millions', 'reporting_year']],
            column_config={
                'facility_name': 'Facility Name',
                'state': 'State',
                'industry_sector_clean': 'Sector',
                'emissions_millions': st.column_config.NumberColumn(
                    'Emissions (M MT)',
                    format="%.2f"
                ),
                'reporting_year': 'Year'
            }
        )
    
    with col2:
        # Sector statistics
        st.subheader("Sector Statistics")
        sector_stats = df.groupby('industry_sector_clean')['total_ghg_emissions_tonnes'].agg([
            'count', 'sum', 'mean', 'std'
        ]).round(2).reset_index()
        sector_stats.columns = ['Sector', 'Count', 'Total', 'Mean', 'Std']
        sector_stats['Total (M MT)'] = sector_stats['Total'] / 1_000_000
        sector_stats['Mean (K MT)'] = sector_stats['Mean'] / 1_000
        
        st.dataframe(
            sector_stats[['Sector', 'Count', 'Total (M MT)', 'Mean (K MT)']].sort_values('Total (M MT)', ascending=False),
            use_container_width=True
        )

def show_emissions_predictor(df, predictor):
    st.markdown('<h2 class="section-header">🔮 Emissions Predictor</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Use the machine learning model to predict GHG emissions for a facility based on its characteristics.
    The model was trained on EPA data from 2021-2023 using a Random Forest algorithm.
    """)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state = st.selectbox(
            "Select State:",
            sorted(df['state'].unique()),
            help="Choose the state where the facility is located"
        )
    
    with col2:
        sector = st.selectbox(
            "Select Industry Sector:",
            sorted(df['industry_sector_clean'].unique()),
            help="Choose the industry sector of the facility"
        )
    
    with col3:
        year = st.selectbox(
            "Select Year:",
            [2021, 2022, 2023, 2024, 2025],
            index=2,
            help="Choose the reporting year"
        )
    
    # Prediction button
    if st.button("Predict Emissions", type="primary"):
        with st.spinner('Making prediction...'):
            try:
                # Create input DataFrame for prediction
                input_data = pd.DataFrame({
                    'state': [state],
                    'industry_sector_clean': [sector],
                    'reporting_year': [year]
                })
                prediction = predictor.predict(input_data)[0]
                
                # Display prediction
                st.success("Prediction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicted Emissions",
                        f"{prediction:,.0f} metric tons CO₂e",
                        help="Predicted annual greenhouse gas emissions"
                    )
                
                with col2:
                    # Find similar facilities for context
                    similar = df[
                        (df['state'] == state) & 
                        (df['industry_sector_clean'] == sector)
                    ]['total_ghg_emissions_tonnes']
                    
                    if len(similar) > 0:
                        avg_similar = similar.mean()
                        st.metric(
                            "Similar Facilities Average",
                            f"{avg_similar:,.0f} metric tons CO₂e",
                            delta=f"{prediction - avg_similar:,.0f}",
                            help="Average emissions of similar facilities in the dataset"
                        )
                
                # Additional context
                st.info(f"""
                **Prediction Context:**
                - **Facility Type:** {sector} facility in {state}
                - **Prediction Year:** {year}
                - **Model:** Random Forest Regressor
                - **Similar Facilities in Dataset:** {len(similar)} facilities
                """)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Show feature importance
    if st.expander("🔍 View Model Feature Importance"):
        importance = predictor.get_feature_importance()
        if importance is not None:
            st.bar_chart(importance.set_index('feature')['importance'])
        else:
            st.info("Feature importance not available for this model type.")

def show_model_performance(training_results, predictor):
    st.markdown('<h2 class="section-header">🤖 Model Performance</h2>', 
                unsafe_allow_html=True)
    
    if training_results is None:
        st.warning("Model performance data not available. Please retrain the model.")
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{training_results['r2']:.4f}")
    with col2:
        st.metric("MAE", f"{training_results['mae']:,.0f} MT")
    with col3:
        st.metric("RMSE", f"{training_results['rmse']:,.0f} MT")
    with col4:
        st.metric("CV Score", f"{training_results['cv_mean']:.4f}")
    
    # Performance visualization
    if 'y_test' in training_results and 'y_pred' in training_results:
        fig_performance = plot_model_performance(
            training_results['y_test'], 
            training_results['y_pred'],
            "Random Forest Model"
        )
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # Feature importance
    if training_results.get('feature_importance') is not None:
        st.subheader("Feature Importance")
        importance_df = training_results['feature_importance']
        st.bar_chart(importance_df.set_index('feature')['importance'])
    
    # Model interpretation
    st.subheader("Model Interpretation")
    st.markdown("""
    **Key Insights:**
    - **R² Score**: Indicates how well the model explains the variance in emissions
    - **MAE (Mean Absolute Error)**: Average prediction error in metric tons
    - **RMSE**: Root mean squared error, penalizes larger errors more heavily
    - **Cross-Validation**: Shows model stability across different data splits
    
    **Feature Importance** shows which factors most influence emissions predictions:
    - Higher importance = stronger influence on predictions
    - Helps identify key drivers of GHG emissions
    """)

if __name__ == "__main__":
    main()