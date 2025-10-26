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

# Custom CSS for shadcn-ui styling with dark/light mode support
st.markdown("""
<style>
    /* Light mode colors */
    :root {
        --background: #ffffff;
        --foreground: #0f172a;
        --card: #ffffff;
        --card-foreground: #0f172a;
        --primary: #1e293b;
        --primary-foreground: #ffffff;
        --secondary: #f1f5f9;
        --secondary-foreground: #0f172a;
        --muted: #f1f5f9;
        --muted-foreground: #64748b;
        --border: #e2e8f0;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --background: #0f172a;
            --foreground: #f8fafc;
            --card: #1e293b;
            --card-foreground: #f8fafc;
            --primary: #f8fafc;
            --primary-foreground: #0f172a;
            --secondary: #334155;
            --secondary-foreground: #f8fafc;
            --muted: #334155;
            --muted-foreground: #94a3b8;
            --border: #334155;
        }
    }

    /* Force dark mode when Streamlit is in dark theme */
    [data-theme="dark"] {
        --background: #0f172a;
        --foreground: #f8fafc;
        --card: #1e293b;
        --card-foreground: #f8fafc;
        --primary: #f8fafc;
        --primary-foreground: #0f172a;
        --secondary: #334155;
        --secondary-foreground: #f8fafc;
        --muted: #334155;
        --muted-foreground: #94a3b8;
        --border: #334155;
    }

    .main {
        padding: 1rem;
        background-color: hsl(var(--background));
        color: hsl(var(--foreground));
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: hsl(var(--foreground)) !important;
    }

    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Update Streamlit components for better theming */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: hsl(var(--foreground)) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: hsl(var(--secondary));
        border: 1px solid hsl(var(--border));
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        color: hsl(var(--secondary-foreground));
    }

    .stTabs [aria-selected="true"] {
        background-color: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        border-color: hsl(var(--primary));
    }

    /* Sidebar styling */
    .css-1d391kg, .css-1544g2n {
        background-color: hsl(var(--card));
        color: hsl(var(--card-foreground));
    }

    /* Metric cards */
    .metric-card {
        background-color: hsl(var(--card));
        color: hsl(var(--card-foreground));
        border: 1px solid hsl(var(--border));
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Data frames and tables */
    .stDataFrame, .stTable {
        background-color: hsl(var(--card));
        color: hsl(var(--card-foreground));
    }

    /* Input elements */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: hsl(var(--card));
        color: hsl(var(--card-foreground));
        border-color: hsl(var(--border));
    }

    /* Buttons */
    .stButton > button {
        background-color: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        border-color: hsl(var(--primary));
    }

    .stButton > button:hover {
        background-color: hsl(var(--primary) / 0.9);
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: hsl(142.1 76.2% 36.3% / 0.1);
        color: hsl(142.1 70.6% 45.3%);
        border-color: hsl(142.1 76.2% 36.3%);
    }

    .stError {
        background-color: hsl(var(--destructive) / 0.1);
        color: hsl(var(--destructive));
        border-color: hsl(var(--destructive));
    }

    .stInfo {
        background-color: hsl(var(--primary) / 0.1);
        color: hsl(var(--primary));
        border-color: hsl(var(--primary));
    }
</style>

<script>
    // Theme detection and sync with Streamlit
    function updateTheme() {
        const streamlitTheme = window.parent.document.querySelector('[data-testid="stApp"]');
        const isDark = streamlitTheme && streamlitTheme.getAttribute('data-theme') === 'dark';
        
        if (isDark) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
    }

    // Update theme on load
    updateTheme();

    // Watch for theme changes
    const observer = new MutationObserver(updateTheme);
    const streamlitApp = window.parent.document.querySelector('[data-testid="stApp"]');
    if (streamlitApp) {
        observer.observe(streamlitApp, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
    }
</script>
""", unsafe_allow_html=True)

def detect_theme():
    """Detect if Streamlit is in dark mode"""
    # Check if the user has set a theme preference
    try:
        # This will work if the theme is set in Streamlit config
        return st.get_option('theme.base') == 'dark'
    except:
        # Fallback to system preference detection
        return False

def get_theme_colors():
    """Get theme-appropriate colors"""
    is_dark = detect_theme()
    
    if is_dark:
        return {
            'background': 'hsl(222.2 84% 4.9%)',
            'foreground': 'hsl(210 40% 98%)',
            'primary': 'hsl(210 40% 98%)',
            'secondary': 'hsl(217.2 32.6% 17.5%)',
            'muted': 'hsl(215 20.2% 65.1%)',
            'border': 'hsl(217.2 32.6% 17.5%)'
        }
    else:
        return {
            'background': 'hsl(0 0% 100%)',
            'foreground': 'hsl(222.2 84% 4.9%)',
            'primary': 'hsl(222.2 47.4% 11.2%)',
            'secondary': 'hsl(210 40% 96%)',
            'muted': 'hsl(215.4 16.3% 46.9%)',
            'border': 'hsl(214.3 31.8% 91.4%)'
        }

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
    # Theme toggle in sidebar
    with st.sidebar:
        st.markdown("### Settings")
        theme_toggle = ui.switch(
            default_checked=detect_theme(),
            label="Dark Mode",
            key="theme_toggle"
        )
        
        if theme_toggle != detect_theme():
            st.rerun()
    
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
    colors = get_theme_colors()
    st.markdown(f'<div class="section-title" style="color: {colors["foreground"]}">Dashboard Overview</div>', unsafe_allow_html=True)
    
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
    colors = get_theme_colors()
    st.markdown(f'<div class="section-title" style="color: {colors["foreground"]}">Detailed Analysis</div>', unsafe_allow_html=True)
    
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
    colors = get_theme_colors()
    st.markdown(f'<div class="section-title" style="color: {colors["foreground"]}">Emissions Predictor</div>', unsafe_allow_html=True)
    
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
    colors = get_theme_colors()
    st.markdown(f'<div class="section-title" style="color: {colors["foreground"]}">Model Performance</div>', unsafe_allow_html=True)
    
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