import json
import urllib.request
import ssl
import pandas as pd
import streamlit as st

CITIES = {
    "Bengaluru, India": {"lat": 12.9716, "lon": 77.5946},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777},
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090},
    "Hyderabad, India": {"lat": 17.3850, "lon": 78.4867},
    "Chennai, India": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata, India": {"lat": 22.5726, "lon": 88.3639},
    "Pune, India": {"lat": 18.5204, "lon": 73.8567},
    "London, UK": {"lat": 51.5074, "lon": -0.1278},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060},
}


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_weather_cached(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&minutely_15=temperature_2m,cloud_cover,shortwave_radiation_instant"
        f"&forecast_days=2&timezone=auto"
    )
    headers = {"User-Agent": "GridWiseAI/1.0"}
    data = None
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        unverified_ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=unverified_ctx) as resp:
            data = json.loads(resp.read().decode())

    if data and "minutely_15" in data:
        df = pd.DataFrame(data["minutely_15"])
        df["timestamp"] = pd.to_datetime(df["time"])
        df = df.rename(
            columns={
                "temperature_2m": "temperature",
                "shortwave_radiation_instant": "solar_irradiance",
            }
        )
        return df
    raise RuntimeError("Incomplete weather data received")


class WeatherService:
    """Service client for external weather telemetries."""

    @staticmethod
    def fetch_live_weather(lat, lon):
        try:
            return _fetch_live_weather_cached(lat, lon)
        except Exception:
            return None

    @staticmethod
    def clear_cache():
        _fetch_live_weather_cached.clear()
