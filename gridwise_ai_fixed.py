"""
GridWise AI - AI-Driven Grid Load Forecasting & Demand Response Platform
Architecture: MVC (Model-View-Controller)
Entry Point: gridwise_ai_fixed.py
"""
import os
import sys
import streamlit as st

# Configure python path for modular MVC imports
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from models.telemetry_store import TelemetryStore
from models.weather_model import CITIES
from models.ml_forecaster import MLForecaster
from controllers.grid_controller import GridController
from views.styles import inject_custom_styles
from views.dashboard_view import DashboardView


def main():
    # ---------------------------------------------------------
    # 1. VIEW INITIALIZATION
    # ---------------------------------------------------------
    st.set_page_config(page_title="AI Grid Load & DR Platform", layout="wide")
    inject_custom_styles()

    # ---------------------------------------------------------
    # 2. MODEL: Train / Load AI Forecaster
    # ---------------------------------------------------------
    train_df = TelemetryStore.generate_training_data()
    model_load, model_solar, features, ai_metrics = MLForecaster.train_models(train_df)

    # ---------------------------------------------------------
    # 3. VIEW: Render Sidebar Controls & User Inputs
    # ---------------------------------------------------------
    (
        selected_city,
        weather_mode,
        transformer_limit,
        battery_capacity,
        weather_cond,
        missing_data_pct,
        live_w_df,
    ) = DashboardView.render_sidebar(CITIES, ai_metrics)

    # ---------------------------------------------------------
    # 4. CONTROLLER: Run Grid Forecast & Mitigation Simulation
    # ---------------------------------------------------------
    # Step 4a: Construct 24-hr Lookahead Horizon
    fut_df = GridController.prepare_horizon(weather_mode, live_w_df, weather_cond)

    # Step 4b: Model Inference (Gradient Boosting Regressors)
    pred_load, pred_solar = MLForecaster.predict(fut_df, model_load, model_solar, features)
    fut_df["pred_load_kw"] = GridController.simulate_meter_pipeline(pred_load, missing_data_pct)
    fut_df["pred_solar_kw"] = pred_solar

    # Step 4c: Physical Mitigation & AI Battery Longevity Analysis
    sim_result = GridController.run_mitigation_simulation(
        fut_df=fut_df,
        transformer_limit=transformer_limit,
        battery_capacity=battery_capacity,
        missing_data_pct=missing_data_pct,
    )

    # ---------------------------------------------------------
    # 5. VIEW: Render Dashboard Presentation
    # ---------------------------------------------------------
    DashboardView.render_header_and_banner(sim_result)
    DashboardView.render_kpi_cards(sim_result)

    col_chart, col_actions = st.columns([3, 1.2])
    with col_chart:
        DashboardView.render_forecast_chart(sim_result, selected_city, weather_mode)
    with col_actions:
        DashboardView.render_automated_mitigation_panel(sim_result)


if __name__ == "__main__":
    main()