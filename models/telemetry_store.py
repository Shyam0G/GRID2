import os
import pandas as pd
import numpy as np
import streamlit as st

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gridwise_dataset.csv")

class TelemetryStore:
    """Handles raw smart meter data, dataset loading, and training data synthesis."""

    @staticmethod
    @st.cache_data
    def load_dataset():
        """Loads historical feeder and smart meter telemetry."""
        if os.path.exists(DATASET_PATH):
            return pd.read_csv(DATASET_PATH)
        return pd.read_csv("gridwise_dataset.csv")

    @staticmethod
    @st.cache_data
    def generate_training_data(n_days=30):
        """Generates realistic historical multi-day telemetry for AI model training."""
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n_days * 24, freq="h")
        df = pd.DataFrame({"timestamp": dates})

        # Calendar and cyclical time features
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Weather profiles
        df["temperature"] = 18 + 10 * np.sin((df["hour"] - 8) * np.pi / 12) + np.random.normal(0, 1.5, len(df))
        df["cloud_cover"] = np.clip(np.random.beta(2, 5, len(df)) * 100 + np.random.normal(0, 5, len(df)), 0, 100)

        sun_profile = np.maximum(0, np.sin((df["hour"] - 6) * np.pi / 13))
        df["solar_irradiance"] = sun_profile * 1000 * (1 - df["cloud_cover"] / 130) + np.random.normal(0, 20, len(df))
        df["solar_irradiance"] = np.clip(df["solar_irradiance"], 0, 1100)

        # Physical rooftop solar generation (kW)
        df["solar_gen_kw"] = df["solar_irradiance"] * 0.12 * (1 - 0.004 * (df["temperature"] - 25))
        df["solar_gen_kw"] = np.clip(df["solar_gen_kw"], 0, None)

        # Gross feeder demand profile (kW)
        morning_peak = np.exp(-((df["hour"] - 9) ** 2) / 8) * 80
        evening_peak = np.exp(-((df["hour"] - 19) ** 2) / 10) * 120
        base_load = 90 + (df["temperature"] - 18) * 2.5
        df["gross_load_kw"] = base_load + morning_peak + evening_peak + np.random.normal(0, 5, len(df))

        return df
