import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_top_sectors(df, top_n=10):
    """Bar chart of top N emitting sectors"""
    sector_emissions = df.groupby('industry_sector_clean')['total_ghg_emissions_tonnes'].agg(['sum', 'count', 'mean']).reset_index()
    sector_emissions = sector_emissions.sort_values('sum', ascending=False).head(top_n)
    
    # Format emissions in millions for better readability
    sector_emissions['sum_millions'] = sector_emissions['sum'] / 1_000_000
    
    fig = px.bar(sector_emissions, 
                 x='sum_millions', 
                 y='industry_sector_clean',
                 title=f'Top {top_n} GHG Emitting Industry Sectors (2021-2023)',
                 labels={'sum_millions': 'Total Emissions (Million Metric Tons)',
                        'industry_sector_clean': 'Industry Sector'},
                 orientation='h',
                 color='sum_millions',
                 color_continuous_scale='Reds')
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder':'total ascending'},
        showlegend=False
    )
    
    return fig

def plot_state_map(df):
    """Choropleth map of emissions by state"""
    state_emissions = df.groupby('state')['total_ghg_emissions_tonnes'].agg(['sum', 'count', 'mean']).reset_index()
    state_emissions['sum_millions'] = state_emissions['sum'] / 1_000_000
    
    fig = px.choropleth(state_emissions,
                        locations='state',
                        locationmode='USA-states',
                        color='sum_millions',
                        scope='usa',
                        title='Total GHG Emissions by State (2021-2023)',
                        labels={'sum_millions': 'Total Emissions (Million Metric Tons)'},
                        color_continuous_scale='Reds',
                        hover_data={'count': True, 'mean': ':.0f'})
    
    fig.update_layout(
        geo=dict(showlakes=True, lakecolor='lightblue'),
        height=500
    )
    
    return fig

def plot_yearly_trends(df):
    """Comprehensive yearly trends analysis"""
    # Overall trends
    yearly_total = df.groupby('reporting_year')['total_ghg_emissions_tonnes'].agg(['sum', 'count', 'mean']).reset_index()
    
    # Top sectors trends
    top_sectors = df.groupby('industry_sector_clean')['total_ghg_emissions_tonnes'].sum().nlargest(5).index
    sector_yearly = df[df['industry_sector_clean'].isin(top_sectors)].groupby(['reporting_year', 'industry_sector_clean'])['total_ghg_emissions_tonnes'].sum().reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Total Emissions Over Time',
            'Average Facility Emissions',
            'Number of Reporting Facilities',
            'Top 5 Sectors Trends'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Total emissions
    fig.add_trace(
        go.Scatter(x=yearly_total['reporting_year'], y=yearly_total['sum']/1_000_000,
                  mode='lines+markers', name='Total Emissions',
                  line=dict(width=3, color='red')),
        row=1, col=1
    )
    
    # Average emissions
    fig.add_trace(
        go.Scatter(x=yearly_total['reporting_year'], y=yearly_total['mean']/1_000,
                  mode='lines+markers', name='Avg Emissions',
                  line=dict(width=3, color='blue')),
        row=1, col=2
    )
    
    # Number of facilities
    fig.add_trace(
        go.Scatter(x=yearly_total['reporting_year'], y=yearly_total['count'],
                  mode='lines+markers', name='Facility Count',
                  line=dict(width=3, color='green')),
        row=2, col=1
    )
    
    # Top sectors
    colors = px.colors.qualitative.Set1
    for i, sector in enumerate(top_sectors):
        sector_data = sector_yearly[sector_yearly['industry_sector_clean'] == sector]
        fig.add_trace(
            go.Scatter(x=sector_data['reporting_year'], y=sector_data['total_ghg_emissions_tonnes']/1_000_000,
                      mode='lines+markers', name=sector[:15],
                      line=dict(width=2, color=colors[i % len(colors)])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="GHG Emissions Trends Analysis (2021-2023)",
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Total Emissions (Million MT)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Emissions (Thousand MT)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Facilities", row=2, col=1)
    fig.update_yaxes(title_text="Emissions (Million MT)", row=2, col=2)
    
    return fig

def plot_outliers(df, threshold_percentile=95):
    """Advanced outlier analysis and visualization"""
    threshold = df['total_ghg_emissions_tonnes'].quantile(threshold_percentile / 100)
    outliers = df[df['total_ghg_emissions_tonnes'] >= threshold].copy()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Emissions Distribution (Log Scale)',
            'Top 20 Outlier Facilities',
            'Outliers by State',
            'Outliers by Sector'
        )
    )
    
    # Log distribution
    fig.add_trace(
        go.Histogram(x=np.log10(df['total_ghg_emissions_tonnes'] + 1),
                    nbinsx=50, name='All Facilities', opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=np.log10(outliers['total_ghg_emissions_tonnes'] + 1),
                    nbinsx=20, name='Outliers', opacity=0.8),
        row=1, col=1
    )
    
    # Top outliers
    top_outliers = outliers.nlargest(20, 'total_ghg_emissions_tonnes')
    fig.add_trace(
        go.Bar(x=top_outliers['total_ghg_emissions_tonnes']/1_000_000,
               y=top_outliers['facility_name'].str[:30],
               orientation='h', name='Top Outliers'),
        row=1, col=2
    )
    
    # Outliers by state
    outliers_by_state = outliers.groupby('state').size().reset_index(name='count')
    outliers_by_state = outliers_by_state.sort_values('count', ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=outliers_by_state['state'], y=outliers_by_state['count'],
               name='Outliers by State'),
        row=2, col=1
    )
    
    # Outliers by sector
    outliers_by_sector = outliers.groupby('industry_sector_clean').size().reset_index(name='count')
    outliers_by_sector = outliers_by_sector.sort_values('count', ascending=False).head(8)
    fig.add_trace(
        go.Bar(x=outliers_by_sector['count'], y=outliers_by_sector['industry_sector_clean'],
               orientation='h', name='Outliers by Sector'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Outlier Analysis (Top {100-threshold_percentile}% of Emitters)",
        showlegend=False
    )
    
    return fig

def plot_model_performance(y_true, y_pred, model_name="Model"):
    """Visualize model performance"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals Plot')
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers',
                  name='Predictions', opacity=0.6),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(dash='dash', color='red')),
        row=1, col=1
    )
    
    # Residuals
    residuals = y_pred - y_true
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers',
                  name='Residuals', opacity=0.6),
        row=1, col=2
    )
    
    # Zero line for residuals
    fig.add_trace(
        go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0],
                  mode='lines', name='Zero Line',
                  line=dict(dash='dash', color='red')),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text=f"{model_name} Performance Analysis",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Actual Emissions", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Emissions", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Emissions", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    
    return fig

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    # Key metrics
    total_emissions = df['total_ghg_emissions_tonnes'].sum()
    avg_emissions = df['total_ghg_emissions_tonnes'].mean()
    total_facilities = df['facility_name'].nunique()
    total_states = df['state'].nunique()
    total_sectors = df['industry_sector_clean'].nunique()
    
    # Create metrics cards data
    metrics = {
        'Total Emissions': f"{total_emissions/1_000_000:.1f}M MT",
        'Average per Facility': f"{avg_emissions/1_000:.1f}K MT",
        'Total Facilities': f"{total_facilities:,}",
        'States Covered': f"{total_states}",
        'Industry Sectors': f"{total_sectors}"
    }
    
    return metrics