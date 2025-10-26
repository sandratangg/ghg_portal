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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for shadcn-ui styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: hsl(222.2 84% 4.9%);
    }
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: hsl(210 40% 98%);
        border: 1px solid hsl(214.3 31.8% 91.4%);
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: hsl(222.2 84% 4.9%);
        color: hsl(210 40% 98%);
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
    # Header with shadcn-ui card
    ui.card(
        content=st.markdown("""
        # GHG Emissions Prediction Portal
        
        Analyze EPA greenhouse gas reporting data from 2021-2023 and predict facility emissions using machine learning.
        This dashboard provides comprehensive insights into emissions patterns across states and industry sectors.
        """),
        key="header_card"
    )
    
    # Load data with progress indicator
    with st.spinner('Loading EPA GHG data...'):
        df = load_data()
    
    # Show data overview with metric cards
    show_data_overview(df)
    
    # Load/train model
    predictor, training_results = load_or_train_model(df)
    
    # Main navigation using tabs
    tabs = ui.tabs(
        options=["Dashboard", "Analysis", "Predictor", "Performance"],
        default_value="Dashboard",
        key="main_navigation"
    )
    
    page = tabs
    
    if page == "Dashboard":
        show_dashboard_overview(df)
    elif page == "Analysis":
        show_detailed_analysis(df)
    elif page == "Predictor":
        show_emissions_predictor(df, predictor)
    elif page == "Performance":
        show_model_performance(training_results, predictor)

def show_data_overview(df):
    """Display data overview with metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.metric_card(
            title="Total Facilities",
            content=f"{len(df):,}",
            description="EPA reporting facilities",
            key="metric_facilities"
        )
    
    with col2:
        ui.metric_card(
            title="Industry Sectors",
            content=f"{df['industry_sector_clean'].nunique()}",
            description="Different sectors",
            key="metric_sectors"
        )
    
    with col3:
        ui.metric_card(
            title="States Covered",
            content=f"{df['state'].nunique()}",
            description="US states/territories",
            key="metric_states"
        )
    
    with col4:
        ui.metric_card(
            title="Total Emissions",
            content=f"{df['total_ghg_emissions_tonnes'].sum()/1e6:.1f}M",
            description="Metric tons CO₂e",
            key="metric_emissions"
        )

def show_dashboard_overview(df):
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Key visualizations in cards
    col1, col2 = st.columns(2)
    
    with col1:
        ui.card(
            content=st.plotly_chart(plot_top_sectors(df), use_container_width=True),
            title="Top Emitting Sectors",
            key="sectors_card"
        )
    
    with col2:
        ui.card(
            content=st.plotly_chart(plot_yearly_trends(df), use_container_width=True),
            title="Emission Trends Over Time",
            key="trends_card"
        )
    
    # State map in full width card
    ui.card(
        content=st.plotly_chart(plot_state_map(df), use_container_width=True),
        title="Emissions by State",
        key="map_card"
    )
    
def show_detailed_analysis(df):
    st.markdown('<div class="section-title">Detailed Analysis</div>', unsafe_allow_html=True)
    
    # Analysis tabs
    analysis_tabs = ui.tabs(
        options=["Outliers", "Sectors", "States", "Time Trends"],
        default_value="Outliers",
        key="analysis_tabs"
    )
    
    if analysis_tabs == "Outliers":
        ui.card(
            content=show_outlier_analysis(df),
            title="Outlier Analysis",
            key="outliers_analysis_card"
        )
    
    elif analysis_tabs == "Sectors":
        ui.card(
            content=show_sector_analysis(df),
            title="Industry Sector Analysis",
            key="sector_analysis_card"
        )
    
    elif analysis_tabs == "States":
        ui.card(
            content=show_state_analysis(df),
            title="State-by-State Analysis",
            key="state_analysis_card"
        )
    
    elif analysis_tabs == "Time Trends":
        ui.card(
            content=show_time_analysis(df),
            title="Temporal Analysis",
            key="time_analysis_card"
        )

def show_outlier_analysis(df):
    """Show outlier analysis with controls"""
    threshold = st.slider(
        "Outlier Threshold (Percentile)",
        min_value=90,
        max_value=99,
        value=95,
        key="outlier_threshold"
    )
    
    fig_outliers = plot_outliers(df, threshold_percentile=threshold)
    st.plotly_chart(fig_outliers, use_container_width=True)
    
    # Show outlier statistics
    outlier_count = len(df[df['total_ghg_emissions_tonnes'] > df['total_ghg_emissions_tonnes'].quantile(threshold/100)])
    st.markdown(f"**Outliers detected:** {outlier_count} facilities ({outlier_count/len(df)*100:.1f}%)")

def show_sector_analysis(df):
    """Show detailed sector analysis"""
    st.plotly_chart(plot_top_sectors(df, top_n=15), use_container_width=True)
    
    # Sector selection
    selected_sectors = st.multiselect(
        "Select sectors to compare:",
        sorted(df['industry_sector_clean'].unique()),
        default=sorted(df['industry_sector_clean'].unique())[:5],
        key="sector_multiselect"
    )
    
    if selected_sectors:
        sector_df = df[df['industry_sector_clean'].isin(selected_sectors)]
        sector_stats = sector_df.groupby('industry_sector_clean')['total_ghg_emissions_tonnes'].agg(['count', 'mean', 'sum']).round(2)
        st.dataframe(sector_stats, use_container_width=True)

def show_state_analysis(df):
    """Show detailed state analysis"""
    st.plotly_chart(plot_state_map(df), use_container_width=True)

def show_time_analysis(df):
    """Show temporal analysis"""
    st.plotly_chart(plot_yearly_trends(df), use_container_width=True)
    
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
    st.markdown('<div class="section-title">Emissions Predictor</div>', unsafe_allow_html=True)
    
    ui.alert(
        title="Machine Learning Prediction",
        description="Use the trained Random Forest model to predict GHG emissions for a facility based on its characteristics. The model was trained on EPA data from 2021-2023.",
        key="predictor_info"
    )
    
    # Input form in a card
    ui.card(
        content=show_prediction_form(df, predictor),
        title="Facility Information",
        key="prediction_form_card"
    )

def show_prediction_form(df, predictor):
    """Show the prediction input form"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state = st.selectbox(
            "Select State:",
            sorted(df['state'].unique()),
            key="state_selector"
        )
    
    with col2:
        sector = st.selectbox(
            "Select Industry Sector:",
            sorted(df['industry_sector_clean'].unique()),
            key="sector_selector"
        )
    
    with col3:
        year = st.selectbox(
            "Select Year:",
            [2021, 2022, 2023, 2024, 2025],
            index=2,
            key="year_selector"
        )
    
    # Prediction button
    if ui.button(text="Predict Emissions", key="predict_button"):
        if state and sector and year:
            make_prediction(df, predictor, state, sector, year)
        else:
            ui.alert(
                title="Missing Information",
                description="Please select all required fields: State, Industry Sector, and Year.",
                key="missing_info_alert"
            )

def make_prediction(df, predictor, state, sector, year):
    """Make and display prediction"""
    try:
        # Create input DataFrame for prediction
        input_data = pd.DataFrame({
            'state': [state],
            'industry_sector_clean': [sector],
            'reporting_year': [year]
        })
        
        prediction = predictor.predict(input_data)[0]
        
        # Display results in cards
        col1, col2 = st.columns(2)
        
        with col1:
            ui.metric_card(
                title="Predicted Emissions",
                content=f"{prediction:,.0f}",
                description="metric tons CO₂e annually",
                key="prediction_result"
            )
        
        with col2:
            # Find similar facilities for context
            similar = df[
                (df['state'] == state) & 
                (df['industry_sector_clean'] == sector)
            ]['total_ghg_emissions_tonnes']
            
            if len(similar) > 0:
                avg_similar = similar.mean()
                delta = prediction - avg_similar
                ui.metric_card(
                    title="Similar Facilities Average",
                    content=f"{avg_similar:,.0f}",
                    description=f"Δ {delta:+,.0f} from average",
                    key="similar_facilities"
                )
        
        # Additional context
        ui.alert(
            title="Prediction Context",
            description=f"Predicted for {sector} facility in {state} for year {year}. Based on {len(similar)} similar facilities in the training dataset.",
            key="prediction_context"
        )
        
    except Exception as e:
        ui.alert(
            title="Prediction Error",
            description=f"An error occurred while making the prediction: {str(e)}",
            key="prediction_error"
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
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    
    if training_results is None:
        ui.alert(
            title="No Performance Data",
            description="Model performance data not available. Please retrain the model to see performance metrics.",
            key="no_performance_alert"
        )
        return
    
    # Performance metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.metric_card(
            title="R² Score",
            content=f"{training_results.get('R2 Score', 0):.4f}",
            description="Model accuracy",
            key="r2_metric"
        )
    
    with col2:
        ui.metric_card(
            title="MAE",
            content=f"{training_results.get('MAE', 0):,.0f}",
            description="Mean Absolute Error (MT)",
            key="mae_metric"
        )
    
    with col3:
        ui.metric_card(
            title="RMSE", 
            content=f"{training_results.get('RMSE', 0):,.0f}",
            description="Root Mean Squared Error (MT)",
            key="rmse_metric"
        )
    
    with col4:
        ui.metric_card(
            title="Cross-Validation",
            content="N/A",
            description="CV Score",
            key="cv_metric"
        )
    
    # Feature importance
    importance = predictor.get_feature_importance()
    if importance is not None:
        ui.card(
            content=st.bar_chart(importance.set_index('feature')['importance']),
            title="Feature Importance",
            key="feature_importance_card"
        )
    
    # Model interpretation
    ui.card(
        content=st.markdown("""
        **Model Insights:**
        
        - **R² Score**: Indicates how well the model explains variance in emissions
        - **MAE**: Average prediction error in metric tons CO₂e
        - **RMSE**: Root mean squared error, penalizes larger errors more heavily
        - **Feature Importance**: Shows which factors most influence emissions predictions
        
        The Random Forest model provides reliable predictions based on state, industry sector, and reporting year.
        """),
        title="Model Interpretation",
        key="interpretation_card"
    )

if __name__ == "__main__":
    main()