import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st


class MLForecaster:
    """AI Machine Learning model suite for feeder demand and solar generation."""

    FEATURE_COLS = [
        "hour", "dayofweek", "is_weekend", "sin_hour", "cos_hour",
        "temperature", "cloud_cover", "solar_irradiance",
    ]

    @staticmethod
    @st.cache_resource
    def train_models(train_df):
        """Trains Gradient Boosting Regressors for gross load and solar forecasting."""
        X = train_df[MLForecaster.FEATURE_COLS]

        # Model 1: AI Feeder Load Predictor
        y_load = train_df["gross_load_kw"]
        model_load = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
        )
        model_load.fit(X, y_load)
        score_load = model_load.score(X, y_load)

        # Model 2: AI Rooftop Solar Predictor
        y_solar = train_df["solar_gen_kw"]
        model_solar = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
        )
        model_solar.fit(X, y_solar)
        score_solar = model_solar.score(X, y_solar)

        metrics = {
            "load_r2": score_load,
            "solar_r2": score_solar,
        }

        return model_load, model_solar, MLForecaster.FEATURE_COLS, metrics

    @staticmethod
    def predict(fut_df, model_load, model_solar, features):
        """Runs inference across future weather and calendar feature horizon."""
        X_fut = fut_df[features]
        pred_load = model_load.predict(X_fut)
        pred_solar = np.clip(model_solar.predict(X_fut), 0, None)

        # Zero out solar generation at night when irradiance is negligible
        mask_night = fut_df["solar_irradiance"] <= 5.0
        pred_solar[mask_night] = 0.0

        return pred_load, pred_solar
