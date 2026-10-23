import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

# ============================================================
# GRIDWISE AI — Distribution Load Forecasting & Dispatch Demo
# ============================================================
# This application uses SYNTHETIC data by default. It is a
# decision-support prototype, not a live grid controller.
# Validate all assumptions against utility data before real use.
# ============================================================

st.set_page_config(
    page_title="GridWise AI - Substation Cockpit",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------- Styling -----------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #0d1117; color: #c9d1d9; }
      [data-testid="stSidebar"] { background-color: #111820; }
      .kpi-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 9px; padding: 14px; min-height: 105px;
      }
      .kpi-label { color: #9da7b3; font-size: 0.82rem; }
      .kpi-value { color: #f0f6fc; font-size: 1.65rem; font-weight: 700; }
      .alert-red {
        background: #481b1b; border-left: 5px solid #ff5555;
        padding: 12px; border-radius: 5px; color: #fff; font-weight: 600;
      }
      .alert-green {
        background: #153a27; border-left: 5px solid #2ecc71;
        padding: 12px; border-radius: 5px; color: #fff; font-weight: 600;
      }
      .note { color: #9da7b3; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Synthetic data generator -------------
@st.cache_data
def generate_training_data(n_days=90, seed=42):
    """Generate reproducible 15-minute synthetic load/weather/solar data."""
    rng = np.random.default_rng(seed)
    periods = n_days * 96
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("15min"),
        periods=periods,
        freq="15min",
    )
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    daily_temp = 27 + 7 * np.sin((df["hour"] - 8) * np.pi / 12)
    weekend_temp_shift = np.zeros(periods)
    df["temperature"] = daily_temp + rng.normal(0, 1.3, periods)
    df["cloud_cover"] = np.clip(rng.beta(2, 5, periods) * 100, 0, 100)

    # Approximate solar profile; zero at night.
    sun_profile = np.maximum(0, np.sin((df["hour"] - 6) * np.pi / 12.5))
    df["solar_irradiance"] = (
        sun_profile * 950 * (1 - df["cloud_cover"] / 125)
    ).clip(lower=0)
    temp_factor = np.clip(1 - 0.004 * (df["temperature"] - 25), 0.75, 1.1)
    df["solar_gen_kw"] = np.maximum(
        0, df["solar_irradiance"] * 0.12 * temp_factor
        + rng.normal(0, 1.8, periods)
    )
    df.loc[df["solar_irradiance"] <= 0, "solar_gen_kw"] = 0

    morning_peak = np.exp(-((df["hour"] - 8.5) ** 2) / 3.5) * 50
    evening_peak = np.exp(-((df["hour"] - 20.0) ** 2) / 5.0) * 85
    weekend_factor = np.where(df["is_weekend"].to_numpy() == 1, 0.92, 1.0)
    temperature_load = np.maximum(0, df["temperature"] - 20) * 2.0
    base_load = 55 + temperature_load
    df["gross_load_kw"] = (
        (base_load + morning_peak + evening_peak) * weekend_factor
        + rng.normal(0, 2.5, periods)
    ).clip(lower=10)

    return df


# --------------------------- Models ---------------------------
LOAD_FEATURES = [
    "hour", "dayofweek", "is_weekend", "sin_hour", "cos_hour",
    "sin_dow", "cos_dow", "temperature",
]
SOLAR_FEATURES = [
    "hour", "sin_hour", "cos_hour", "temperature",
    "cloud_cover", "solar_irradiance",
]


@st.cache_resource
def train_models(data_signature, train_df):
    """Train models and evaluate on a chronological holdout set."""
    del data_signature  # used as a cache key
    df = train_df.copy().sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    load_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        loss="huber", random_state=42,
    )
    solar_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        loss="huber", random_state=42,
    )
    load_model.fit(train[LOAD_FEATURES], train["gross_load_kw"])
    solar_model.fit(train[SOLAR_FEATURES], train["solar_gen_kw"])

    load_pred = np.maximum(0, load_model.predict(test[LOAD_FEATURES]))
    solar_pred = np.maximum(0, solar_model.predict(test[SOLAR_FEATURES]))
    metrics = {
        "load_mae": mean_absolute_error(test["gross_load_kw"], load_pred),
        "load_rmse": np.sqrt(mean_squared_error(test["gross_load_kw"], load_pred)),
        "solar_mae": mean_absolute_error(test["solar_gen_kw"], solar_pred),
        "solar_rmse": np.sqrt(mean_squared_error(test["solar_gen_kw"], solar_pred)),
        "test_start": test["timestamp"].iloc[0],
        "test_end": test["timestamp"].iloc[-1],
    }
    return load_model, solar_model, metrics


# -------------------- Telemetry / future horizon --------------
def build_forecast_horizon(scenario, packet_loss_pct, seed=123):
    """Create a 24-hour weather forecast and simulate missing weather telemetry."""
    rng = np.random.default_rng(seed)
    horizon = pd.date_range(
        start=pd.Timestamp.now().ceil("15min"),
        periods=96,
        freq="15min",
    )
    df = pd.DataFrame({"timestamp": horizon})
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    if scenario == "Normal Day":
        df["cloud_cover"] = 18 + 8 * np.sin((df["hour"] - 10) * np.pi / 12)
        df["temperature"] = 26 + 7 * np.sin((df["hour"] - 8) * np.pi / 12)
    elif scenario == "Solar Drop (Sudden Thunderstorm)":
        df["cloud_cover"] = np.where(
            (df["hour"] >= 11) & (df["hour"] <= 15), 92.0, 22.0
        )
        df["temperature"] = 24 + 4 * np.sin((df["hour"] - 8) * np.pi / 12)
    else:
        df["cloud_cover"] = 8.0
        df["temperature"] = 35 + 8 * np.sin((df["hour"] - 8) * np.pi / 12)

    df["cloud_cover"] = df["cloud_cover"].clip(0, 100)

    # Simulate missing telemetry readings, then impute. This is a demonstration
    # of missing weather inputs, not a claim that future meter readings exist.
    original_temp = df["temperature"].copy()
    original_cloud = df["cloud_cover"].copy()
    missing = rng.random(len(df)) < packet_loss_pct / 100.0
    df.loc[missing, "temperature"] = np.nan
    df.loc[missing, "cloud_cover"] = np.nan
    df["temperature"] = df["temperature"].interpolate(limit_direction="both")
    df["cloud_cover"] = df["cloud_cover"].interpolate(limit_direction="both")
    if df["temperature"].isna().any():
        df["temperature"] = df["temperature"].fillna(original_temp)
    if df["cloud_cover"].isna().any():
        df["cloud_cover"] = df["cloud_cover"].fillna(original_cloud)

    sun_profile = np.maximum(0, np.sin((df["hour"] - 6) * np.pi / 12.5))
    df["solar_irradiance"] = (
        sun_profile * 950 * (1 - df["cloud_cover"] / 125)
    ).clip(lower=0)
    return df, int(missing.sum())


# ----------------------- Battery dispatch ----------------------
def greedy_dispatch(net_load, safe_limit, capacity_kwh, power_kw,
                    dr_limit_kw, initial_soc_fraction=0.60,
                    min_soc_fraction=0.15, max_soc_fraction=0.95,
                    charge_eff=0.95, discharge_eff=0.95):
    """Feasible fallback schedule. Positive battery power means discharge."""
    dt = 0.25
    soc = initial_soc_fraction * capacity_kwh
    min_soc = min_soc_fraction * capacity_kwh
    max_soc = max_soc_fraction * capacity_kwh
    p_bat = np.zeros(len(net_load))
    p_dr = np.zeros(len(net_load))
    soc_trace = np.zeros(len(net_load))

    for i, load in enumerate(net_load):
        overload = max(0.0, load - safe_limit)
        available_energy_kw = max(0.0, (soc - min_soc) * discharge_eff / dt)
        discharge = min(overload, power_kw, available_energy_kw)
        p_bat[i] = discharge
        soc -= (discharge / discharge_eff) * dt

        remaining = max(0.0, load - discharge - safe_limit)
        p_dr[i] = min(remaining, dr_limit_kw)
        soc_trace[i] = soc

    final_net = net_load - p_bat - p_dr
    return final_net, p_bat, p_dr, soc_trace


def optimize_dispatch(net_load, safe_limit, capacity_kwh, power_kw, dr_limit_kw,
                     initial_soc_fraction=0.60, min_soc_fraction=0.15,
                     max_soc_fraction=0.95, charge_eff=0.95,
                     discharge_eff=0.95):
    """
    Optimize battery dispatch and demand response over a 24-hour horizon.

    Convention:
      p_bat > 0: discharge (reduces grid import)
      p_bat < 0: charge (increases grid import)
      p_dr >= 0: flexible demand curtailed

    SOC dynamics include charge/discharge efficiency. Constraints are applied
    at every interval. If the optimizer fails, use a feasible greedy fallback.
    """
    net_load = np.asarray(net_load, dtype=float)
    n = len(net_load)
    dt = 0.25
    min_soc = min_soc_fraction * capacity_kwh
    max_soc = max_soc_fraction * capacity_kwh
    initial_soc = np.clip(initial_soc_fraction * capacity_kwh, min_soc, max_soc)

    if capacity_kwh <= 0 or power_kw <= 0:
        final_net, p_bat, p_dr, soc_trace = greedy_dispatch(
            net_load, safe_limit, capacity_kwh, power_kw, dr_limit_kw,
            initial_soc_fraction, min_soc_fraction, max_soc_fraction,
            charge_eff, discharge_eff
        )
        return final_net, p_bat, p_dr, soc_trace, "Battery unavailable; fallback dispatch"

    def soc_from_power(p):
        # For positive p, remove p/eta energy. For negative p, add -p*eta.
        energy_rate = np.where(p >= 0, p / discharge_eff, p * charge_eff)
        return initial_soc - np.cumsum(energy_rate) * dt

    def objective(x):
        p = x[:n]
        dr = x[n:]
        after = net_load - p - dr
        overload = np.maximum(after - safe_limit, 0)
        # Strong overload penalty; explicit cost for curtailment and battery wear.
        overload_cost = 1500.0 * np.sum(overload ** 2)
        dr_cost = 30.0 * np.sum(dr) * dt
        battery_wear_cost = 0.08 * np.sum(p ** 2) * dt
        peak_cost = 15.0 * max(0.0, np.max(after) - safe_limit) ** 2
        return overload_cost + dr_cost + battery_wear_cost + peak_cost

    def soc_lower(x):
        soc = soc_from_power(x[:n])
        return soc - min_soc

    def soc_upper(x):
        soc = soc_from_power(x[:n])
        return max_soc - soc

    # Start at zero dispatch. SLSQP is given physically valid bounds.
    x0 = np.zeros(2 * n)
    bounds = [(-power_kw, power_kw)] * n + [(0.0, max(0.0, dr_limit_kw))] * n
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": soc_lower},
            {"type": "ineq", "fun": soc_upper},
        ],
        options={"maxiter": 250, "ftol": 1e-5, "disp": False},
    )

    if result.success and np.all(np.isfinite(result.x)):
        p_bat = result.x[:n]
        p_dr = result.x[n:]
        soc_trace = soc_from_power(p_bat)
        final_net = net_load - p_bat - p_dr
        # Numerical tolerance guard: if the result violates SOC bounds, fallback.
        if (np.min(soc_trace) >= min_soc - 1e-3
                and np.max(soc_trace) <= max_soc + 1e-3):
            return final_net, p_bat, p_dr, soc_trace, "SLSQP optimization successful"

    final_net, p_bat, p_dr, soc_trace = greedy_dispatch(
        net_load, safe_limit, capacity_kwh, power_kw, dr_limit_kw,
        initial_soc_fraction, min_soc_fraction, max_soc_fraction,
        charge_eff, discharge_eff
    )
    return final_net, p_bat, p_dr, soc_trace, "Optimizer fallback used"


# --------------------------- Helpers ---------------------------
def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def make_dark_layout(fig, height=420):
    fig.update_layout(
        plot_bgcolor="#161b22",
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=25),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        xaxis=dict(gridcolor="#30363d", nticks=12),
        yaxis=dict(gridcolor="#30363d"),
    )
    return fig


# ------------------------------ App ----------------------------
def run_app():
    st.title("⚡ GRIDWISE AI — Substation Transformer Cockpit")
    st.caption(
        "Load and rooftop-solar forecasting • Battery dispatch • Demand response • "
        "Synthetic-data prototype"
    )
    st.warning(
        "Prototype notice: this demo uses synthetic training data and simplified "
        "operating assumptions. It is not connected to a live utility grid and "
        "must not directly control real equipment."
    )

    data = generate_training_data()
    signature = f"{len(data)}-{data['timestamp'].iloc[0]}-{data['timestamp'].iloc[-1]}"
    load_model, solar_model, metrics = train_models(signature, data)

    # -------------------------- Controls ------------------------
    st.sidebar.header("⚙️ Feeder & Battery")
    transformer_limit = st.sidebar.slider(
        "Operating limit (kW)", 80, 200, 115, 5
    )
    bess_energy = st.sidebar.slider(
        "Battery capacity (kWh)", 0, 300, 60, 10
    )
    bess_power = st.sidebar.slider(
        "Battery inverter rating (kW)", 0, 100, 25, 5
    )
    initial_soc_pct = st.sidebar.slider("Initial battery SoC (%)", 15, 95, 60, 5)
    min_soc_pct = st.sidebar.slider("Minimum battery SoC (%)", 5, 40, 15, 5)
    max_soc_pct = st.sidebar.slider("Maximum battery SoC (%)", 60, 100, 95, 5)
    if min_soc_pct >= max_soc_pct:
        st.sidebar.error("Maximum SoC must be greater than minimum SoC.")
        st.stop()
    if not (min_soc_pct <= initial_soc_pct <= max_soc_pct):
        st.sidebar.warning("Initial SoC is clipped to the configured SoC limits.")

    st.sidebar.markdown("---")
    st.sidebar.header("🌦️ Scenario & Data Quality")
    scenario = st.sidebar.selectbox(
        "Stress scenario",
        ["Normal Day", "Solar Drop (Sudden Thunderstorm)", "Evening Heatwave Surge"],
    )
    packet_loss = st.sidebar.slider("Simulated weather-input packet loss (%)", 0, 40, 10, 5)
    dr_limit = st.sidebar.slider("Maximum demand response (kW)", 0, 50, 15, 5)
    st.sidebar.caption("Packet loss here applies to forecast weather inputs, not actual AMI meter history.")

    # ---------------------- Forecast pipeline ------------------
    fut_df, missing_count = build_forecast_horizon(scenario, packet_loss)
    fut_df["solar_irradiance"] = (
        fut_df["solar_irradiance"] *
        np.clip(1 - 0.004 * (fut_df["temperature"] - 25), 0.75, 1.1)
    ).clip(lower=0)

    fut_df["pred_load_kw"] = np.maximum(
        0, load_model.predict(fut_df[LOAD_FEATURES])
    )
    fut_df["pred_solar_kw"] = np.maximum(
        0, solar_model.predict(fut_df[SOLAR_FEATURES])
    )
    # Solar is physically zero at night regardless of model noise.
    fut_df.loc[fut_df["solar_irradiance"] <= 0, "pred_solar_kw"] = 0.0
    fut_df["net_load_kw"] = fut_df["pred_load_kw"] - fut_df["pred_solar_kw"]

    mitigated, p_bat, p_dr, soc_trace, optimizer_status = optimize_dispatch(
        fut_df["net_load_kw"].to_numpy(),
        transformer_limit,
        bess_energy,
        bess_power,
        dr_limit,
        initial_soc_fraction=initial_soc_pct / 100,
        min_soc_fraction=min_soc_pct / 100,
        max_soc_fraction=max_soc_pct / 100,
    )
    fut_df["battery_dispatch_kw"] = p_bat
    fut_df["dr_kw"] = p_dr
    fut_df["mitigated_net_kw"] = mitigated
    fut_df["soc_kwh"] = soc_trace
    fut_df["overload_before_kw"] = np.maximum(fut_df["net_load_kw"] - transformer_limit, 0)
    fut_df["overload_after_kw"] = np.maximum(fut_df["mitigated_net_kw"] - transformer_limit, 0)

    # ----------------------- Key metrics ------------------------
    dt = 0.25
    peak_before = float(fut_df["net_load_kw"].max())
    peak_after = float(fut_df["mitigated_net_kw"].max())
    peak_reduction = max(0.0, peak_before - peak_after)
    overload_before = float(fut_df["overload_before_kw"].max())
    overload_after = float(fut_df["overload_after_kw"].max())
    overload_intervals_before = int((fut_df["overload_before_kw"] > 1e-3).sum())
    overload_intervals_after = int((fut_df["overload_after_kw"] > 1e-3).sum())
    discharge_kwh = float(np.maximum(p_bat, 0).sum() * dt)
    charge_kwh = float(np.maximum(-p_bat, 0).sum() * dt)
    dr_energy_kwh = float(np.sum(p_dr) * dt)
    solar_energy_kwh = float(fut_df["pred_solar_kw"].sum() * dt)
    min_soc_seen = float(np.min(soc_trace)) if len(soc_trace) else 0.0
    max_soc_seen = float(np.max(soc_trace)) if len(soc_trace) else 0.0
    peak_time = fut_df.loc[fut_df["net_load_kw"].idxmax(), "timestamp"]
    peak_after_time = fut_df.loc[fut_df["mitigated_net_kw"].idxmax(), "timestamp"]

    # Illustrative energy-arbitrage calculation, separate from demand-charge impact.
    # Prices are user-configurable assumptions, not a verified local tariff.
    st.sidebar.markdown("---")
    st.sidebar.header("₹ Illustrative price assumptions")
    peak_price = st.sidebar.number_input("Avoided import price (₹/kWh)", 0.0, 100.0, 10.0, 0.5)
    charge_price = st.sidebar.number_input("Charging energy price (₹/kWh)", 0.0, 100.0, 3.2, 0.5)
    battery_wear = st.sidebar.number_input("Battery wear cost (₹/discharged kWh)", 0.0, 20.0, 1.0, 0.25)
    demand_charge = st.sidebar.number_input(
        "Demand-charge rate (₹/kW per billing period)", 0.0, 5000.0, 450.0, 25.0
    )
    arbitrage_value = (
        discharge_kwh * peak_price
        - charge_kwh * charge_price
        - discharge_kwh * battery_wear
    )
    demand_charge_impact = peak_reduction * demand_charge
    st.sidebar.caption(
        "Demand-charge impact is a potential billing-period value, not a daily saving. "
        "Confirm tariff rules and billing-period peak definitions before using it."
    )

    if peak_before > transformer_limit:
        st.markdown(
            f'<div class="alert-red">🚨 FORECAST RISK: peak net load is '
            f'{peak_before:.1f} kW, {peak_before - transformer_limit:.1f} kW above '
            f'the configured operating limit. Dispatch optimization has been evaluated.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-green">✅ FORECAST WITHIN LIMIT: the predicted net load '
            'remains within the configured operating limit before mitigation.</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Optimization status: {optimizer_status} • Forecast horizon: 24 hours • "
        f"15-minute intervals • Missing simulated weather inputs: {missing_count}/96"
    )

    tab_cockpit, tab_dispatch, tab_xai, tab_validation = st.tabs(
        ["📊 Operational Cockpit", "🛡️ Dispatch & Economics",
         "🧠 Explainable AI", "🧪 Model Validation"]
    )

    # --------------------------- Cockpit ------------------------
    with tab_cockpit:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("FORECAST PEAK", f"{peak_before:.1f} kW",
                        f"At {peak_time.strftime('%H:%M')}")
        with k2:
            metric_card("PEAK AFTER DISPATCH", f"{peak_after:.1f} kW",
                        f"At {peak_after_time.strftime('%H:%M')}")
        with k3:
            metric_card("PEAK REDUCTION", f"{peak_reduction:.1f} kW",
                        "Forecast baseline minus optimized peak")
        with k4:
            metric_card("SOLAR ENERGY", f"{solar_energy_kwh:.1f} kWh",
                        "Forecast energy over 24 hours")

        fig = go.Figure()
        times = fut_df["timestamp"]
        fig.add_trace(go.Scatter(x=times, y=fut_df["pred_load_kw"], name="Gross demand",
                                 line=dict(color="#00d4ff", width=2)))
        fig.add_trace(go.Scatter(x=times, y=fut_df["pred_solar_kw"], name="Solar generation",
                                 line=dict(color="#f1c40f", width=2)))
        fig.add_trace(go.Scatter(x=times, y=fut_df["net_load_kw"], name="Net load before dispatch",
                                 line=dict(color="#b58cff", width=2)))
        fig.add_trace(go.Scatter(x=times, y=fut_df["mitigated_net_kw"], name="Net load after dispatch",
                                 line=dict(color="#2ecc71", width=2.5)))
        fig.add_hline(y=transformer_limit, line_dash="dash", line_color="#ff5555",
                      annotation_text="Configured operating limit")
        fig.update_layout(title="24-hour forecast and mitigation schedule",
                          yaxis_title="Power (kW)")
        st.plotly_chart(make_dark_layout(fig, 460), use_container_width=True)

        st.subheader("Forecast intervals")
        display = fut_df[[
            "timestamp", "pred_load_kw", "pred_solar_kw", "net_load_kw",
            "battery_dispatch_kw", "dr_kw", "mitigated_net_kw", "soc_kwh"
        ]].copy()
        display.columns = [
            "Time", "Demand (kW)", "Solar (kW)", "Net before (kW)",
            "Battery (+ discharge / - charge) (kW)", "Demand response (kW)",
            "Net after (kW)", "Battery SoC (kWh)"
        ]
        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download forecast and dispatch CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="gridwise_forecast_dispatch.csv",
            mime="text/csv",
        )

    # ---------------------- Dispatch and economics --------------
    with tab_dispatch:
        a, b, c, d = st.columns(4)
        with a:
            metric_card("MAX OVERLOAD BEFORE", f"{overload_before:.1f} kW",
                        f"{overload_intervals_before} intervals above limit")
        with b:
            metric_card("MAX OVERLOAD AFTER", f"{overload_after:.1f} kW",
                        f"{overload_intervals_after} intervals above limit")
        with c:
            metric_card("BATTERY DISCHARGED", f"{discharge_kwh:.1f} kWh",
                        f"Charged: {charge_kwh:.1f} kWh")
        with d:
            metric_card("DEMAND RESPONSE", f"{dr_energy_kwh:.1f} kWh",
                        "Flexible load curtailed in this simulation")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=times, y=p_bat, name="Battery power (+ discharge, - charge)",
                              marker_color="#00d4ff"))
        fig2.add_trace(go.Scatter(x=times, y=p_dr, name="Demand response",
                                  line=dict(color="#ff9f43", width=2)))
        fig2.update_layout(title="Recommended dispatch by 15-minute interval",
                           yaxis_title="Power (kW)")
        st.plotly_chart(make_dark_layout(fig2, 360), use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=times, y=soc_trace, name="Battery SoC",
                                  line=dict(color="#2ecc71", width=2.5)))
        fig3.add_hline(y=min_soc_pct / 100 * bess_energy, line_dash="dash",
                       line_color="#ff5555", annotation_text="Minimum SoC")
        fig3.add_hline(y=max_soc_pct / 100 * bess_energy, line_dash="dash",
                       line_color="#f1c40f", annotation_text="Maximum SoC")
        fig3.update_layout(title="Battery energy state (efficiency-aware)",
                           yaxis_title="Stored energy (kWh)")
        st.plotly_chart(make_dark_layout(fig3, 330), use_container_width=True)

        st.subheader("Illustrative financial estimates")
        e1, e2 = st.columns(2)
        with e1:
            st.metric("Estimated energy-arbitrage value", f"₹ {arbitrage_value:,.0f}")
            st.caption(
                "Uses the entered energy prices and battery-wear assumption. "
                "Can be negative; excludes taxes, network charges and other tariff details."
            )
        with e2:
            st.metric("Potential demand-charge impact", f"₹ {demand_charge_impact:,.0f}")
            st.caption(
                "Peak reduction × entered ₹/kW billing-period rate. This is not a daily "
                "saving and is not added to arbitrage value."
            )
        if arbitrage_value < 0:
            st.warning(
                "Under the current assumptions, energy arbitrage has a negative estimated "
                "value. Do not force battery cycling just to show a saving."
            )
        st.info(
            "These calculations are scenario estimates, not a verified bill reduction. "
            "Real savings require the actual tariff, billing rules, battery losses, "
            "degradation model and baseline operating strategy."
        )

    # ------------------------ Explainable AI --------------------
    with tab_xai:
        st.subheader("What drives the predictions?")
        st.write(
            "Feature importance measures how much each input contributes to splits "
            "across the fitted tree ensemble. It is a global diagnostic, not a causal "
            "explanation for an individual interval."
        )
        x1, x2 = st.columns(2)
        with x1:
            imp_load = pd.DataFrame({
                "Feature": LOAD_FEATURES,
                "Importance": load_model.feature_importances_,
            }).sort_values("Importance")
            f1 = go.Figure(go.Bar(
                x=imp_load["Importance"], y=imp_load["Feature"], orientation="h",
                marker_color="#00d4ff",
            ))
            f1.update_layout(title="Gross-demand model feature importance",
                             xaxis_title="Relative importance")
            st.plotly_chart(make_dark_layout(f1, 360), use_container_width=True)
        with x2:
            imp_solar = pd.DataFrame({
                "Feature": SOLAR_FEATURES,
                "Importance": solar_model.feature_importances_,
            }).sort_values("Importance")
            f2 = go.Figure(go.Bar(
                x=imp_solar["Importance"], y=imp_solar["Feature"], orientation="h",
                marker_color="#f1c40f",
            ))
            f2.update_layout(title="Solar model feature importance",
                             xaxis_title="Relative importance")
            st.plotly_chart(make_dark_layout(f2, 360), use_container_width=True)

        st.subheader("Why dispatch was selected")
        reasons = []
        if peak_before > transformer_limit:
            reasons.append(
                f"- Forecast peak exceeds the configured limit by "
                f"{peak_before - transformer_limit:.1f} kW."
            )
        else:
            reasons.append("- The baseline forecast is within the configured limit.")
        reasons.append(
            f"- Battery discharge energy scheduled: {discharge_kwh:.1f} kWh; "
            f"charging energy scheduled: {charge_kwh:.1f} kWh."
        )
        reasons.append(
            f"- Demand-response energy scheduled: {dr_energy_kwh:.1f} kWh, "
            f"capped at {dr_limit:.1f} kW per interval."
        )
        reasons.append(
            f"- Simulated minimum/maximum battery energy: {min_soc_seen:.1f}/"
            f"{max_soc_seen:.1f} kWh."
        )
        reasons.append(
            f"- Remaining maximum overload after mitigation: {overload_after:.1f} kW. "
            "If non-zero, the available flexibility was insufficient or the optimizer "
            "could not fully eliminate the peak."
        )
        st.markdown("\n".join(reasons))

    # ----------------------- Model validation -------------------
    with tab_validation:
        st.subheader("Chronological holdout evaluation")
        st.caption(
            f"Models trained on the first 80% of synthetic records and evaluated on "
            f"the final 20% ({metrics['test_start']} to {metrics['test_end']}). "
            "These scores describe this synthetic dataset only, not real-world accuracy."
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("LOAD MAE", f"{metrics['load_mae']:.2f} kW",
                        "Mean absolute error")
        with m2:
            metric_card("LOAD RMSE", f"{metrics['load_rmse']:.2f} kW",
                        "Root mean squared error")
        with m3:
            metric_card("SOLAR MAE", f"{metrics['solar_mae']:.2f} kW",
                        "Mean absolute error")
        with m4:
            metric_card("SOLAR RMSE", f"{metrics['solar_rmse']:.2f} kW",
                        "Root mean squared error")

        st.subheader("Synthetic training data sample")
        st.dataframe(data.tail(30).round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "Download synthetic training data CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="gridwise_synthetic_training_data.csv",
            mime="text/csv",
        )
        st.warning(
            "For a credible field evaluation, replace synthetic data with time-aligned "
            "meter readings and weather observations, then evaluate on a later period "
            "that was never used for training or model selection."
        )


if __name__ == "__main__":
    run_app()
