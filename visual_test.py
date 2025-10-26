import sys
import os

# add data/src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'src'))

from data_processing import load_and_clean_data, prepare_features
from visualizations import (
    plot_top_sectors,
    plot_state_map,
    plot_yearly_trends,
    plot_outliers,
    plot_model_performance,
    create_summary_dashboard,
)

def main():
    df = load_and_clean_data()

    checks = [
        plot_top_sectors,
        plot_state_map,
        plot_yearly_trends,
        plot_outliers,
    ]

    for fn in checks:
        try:
            _ = fn(df)
            print(f"{fn.__name__}: OK")
        except Exception as e:
            print(f"{fn.__name__}: ERROR: {e}")

    # quick performance plot test using sample arrays
    y_true = df['total_ghg_emissions_tonnes'].iloc[:50]
    y_pred = (y_true * 0.9).values
    try:
        _ = plot_model_performance(y_true.reset_index(drop=True), y_pred)
        print("plot_model_performance: OK")
    except Exception as e:
        print(f"plot_model_performance: ERROR: {e}")

    try:
        metrics = create_summary_dashboard(df)
        print("create_summary_dashboard: OK", list(metrics.keys()))
    except Exception as e:
        print(f"create_summary_dashboard: ERROR: {e}")

if __name__ == '__main__':
    main()
