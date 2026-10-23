import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GridSimulationResult:
    """Holds all calculated grid telemetry, forecast trajectories, and mitigation analytics."""
    fut_df: pd.DataFrame
    transformer_limit: float
    battery_capacity: float
    battery_storage_kwh: float
    has_overload: bool
    max_overload: float
    overload_hour: Optional[str]
    peak_reduction_kw: float
    cost_savings_inr: float
    active_dispatch_hours: float
    total_batt_dispatched_kwh: float
    ai_battery_hours: float
    ai_usage_summary: str
    ai_usage_profile_detail: str
    max_batt: float
    max_dr: float
    current_3phase_amps: float
    current_1phase_amps: float
    geysers_count: int
    acs_count: int
    evs_count: int
    missing_data_pct: int


class GridController:
    """Controller orchestrating AI inference, grid physics, and mitigation logic."""

    DT = 0.25  # 15-minute intervals = 0.25 hours

    @staticmethod
    def prepare_horizon(weather_mode: str, live_w_df: Optional[pd.DataFrame], weather_cond: str) -> pd.DataFrame:
        """Constructs 24-hour ahead (96-interval) environmental and calendar feature horizon."""
        if weather_mode == "🌐 Live API (AI Prediction)" and live_w_df is not None:
            fut_df = live_w_df.iloc[:96].copy().reset_index(drop=True)
            fut_df["hour"] = fut_df["timestamp"].dt.hour + fut_df["timestamp"].dt.minute / 60.0
            fut_df["dayofweek"] = fut_df["timestamp"].dt.dayofweek
            fut_df["is_weekend"] = fut_df["dayofweek"].isin([5, 6]).astype(int)
            fut_df["sin_hour"] = np.sin(2 * np.pi * fut_df["hour"] / 24)
            fut_df["cos_hour"] = np.cos(2 * np.pi * fut_df["hour"] / 24)
        else:
            forecast_dates = pd.date_range(start=pd.Timestamp.now().floor("15min"), periods=96, freq="15min")
            fut_df = pd.DataFrame({"timestamp": forecast_dates})
            fut_df["hour"] = fut_df["timestamp"].dt.hour + fut_df["timestamp"].dt.minute / 60.0
            fut_df["dayofweek"] = fut_df["timestamp"].dt.dayofweek
            fut_df["is_weekend"] = fut_df["dayofweek"].isin([5, 6]).astype(int)
            fut_df["sin_hour"] = np.sin(2 * np.pi * fut_df["hour"] / 24)
            fut_df["cos_hour"] = np.cos(2 * np.pi * fut_df["hour"] / 24)

            # Baseline scenario weather features
            fut_df["temperature"] = 20.0 + 8.0 * np.sin((fut_df["hour"] - 8) * np.pi / 12)
            fut_df["cloud_cover"] = 15.0
            sun_profile = np.maximum(0, np.sin((fut_df["hour"] - 6) * np.pi / 13))
            fut_df["solar_irradiance"] = np.clip(sun_profile * 950 * (1 - fut_df["cloud_cover"] / 130), 0, 1100)

            if weather_cond == "Cloudy Afternoon (Solar Drop)":
                cloud_mask = (fut_df["hour"] >= 13) & (fut_df["hour"] <= 18)
                fut_df.loc[cloud_mask, "cloud_cover"] = 90.0
                fut_df.loc[cloud_mask, "solar_irradiance"] = fut_df.loc[cloud_mask, "solar_irradiance"] * 0.15
            elif weather_cond == "Sudden Evening Heatwave":
                heat_mask = (fut_df["hour"] >= 17) & (fut_df["hour"] <= 22)
                fut_df.loc[heat_mask, "temperature"] += 8.5

        return fut_df

    @staticmethod
    def simulate_meter_pipeline(pred_load: np.ndarray, missing_data_pct: int) -> pd.Series:
        """Simulates missing or noisy smart meter sensor data and applies linear interpolation."""
        raw = pd.Series(pred_load)
        if missing_data_pct > 0:
            np.random.seed(42)
            mask = np.random.rand(len(raw)) < (missing_data_pct / 100)
            simulated = raw.copy()
            simulated[mask] = np.nan
            return simulated.interpolate(method="linear").ffill().bfill()
        return raw

    @staticmethod
    def run_mitigation_simulation(
        fut_df: pd.DataFrame,
        transformer_limit: float,
        battery_capacity: float,
        missing_data_pct: int,
        battery_storage_kwh: Optional[float] = None,
    ) -> GridSimulationResult:
        """Calculates physical net load, overload events, battery dispatch, DR and longevity."""
        dt = GridController.DT
        if battery_storage_kwh is None:
            battery_storage_kwh = float(battery_capacity * 2.5)

        # Net load is physical balance: Consumer Demand minus Rooftop Generation
        fut_df["net_load_kw"] = fut_df["pred_load_kw"] - fut_df["pred_solar_kw"]

        # Identify Overload
        fut_df["overload_kw"] = np.maximum(0, fut_df["net_load_kw"] - transformer_limit)
        has_overload = bool(fut_df["overload_kw"].max() > 0)
        max_overload = float(fut_df["overload_kw"].max())
        overload_hour = fut_df.loc[fut_df["overload_kw"].idxmax(), "timestamp"].strftime("%H:%M") if has_overload else None

        # Mitigation: Battery + Demand Response (respecting both power kW and energy kWh limits)
        remaining_energy = float(battery_storage_kwh)
        batt_dispatches = []
        for ov in fut_df["overload_kw"]:
            if ov > 0 and remaining_energy > 0 and battery_capacity > 0:
                max_p = remaining_energy / dt
                p_dispatch = min(float(ov), float(battery_capacity), max_p)
                remaining_energy -= p_dispatch * dt
                batt_dispatches.append(p_dispatch)
            else:
                batt_dispatches.append(0.0)

        fut_df["battery_dispatch_kw"] = batt_dispatches
        fut_df["remaining_overload_kw"] = fut_df["overload_kw"] - fut_df["battery_dispatch_kw"]
        fut_df["demand_response_kw"] = fut_df["remaining_overload_kw"]
        fut_df["mitigated_net_load_kw"] = fut_df["net_load_kw"] - fut_df["battery_dispatch_kw"] - fut_df["demand_response_kw"]

        # Operational Battery Statistics
        active_dispatch_hours = float((fut_df["battery_dispatch_kw"] > 0).sum() * dt)
        total_batt_dispatched_kwh = float(fut_df["battery_dispatch_kw"].sum() * dt)

        # ---------------------------------------------------------
        # AI Dynamic Battery Longevity Analysis (Based on Usage)
        # ---------------------------------------------------------
        if battery_capacity <= 0 or battery_storage_kwh <= 0:
            ai_battery_hours = 0.0
            ai_usage_summary = "No active battery storage connected"
            ai_usage_profile_detail = "Storage capacity is 0"
        elif has_overload:
            ov_mask = fut_df["overload_kw"] > 0
            ov_indices = fut_df.index[ov_mask].tolist()
            total_peak_hours = len(ov_indices) * dt

            dispatched_series = fut_df.loc[ov_mask, "battery_dispatch_kw"]
            total_kwh_discharged = float(dispatched_series.sum() * dt)
            active_intervals = int((dispatched_series > 0).sum())
            active_support_hours = active_intervals * dt
            avg_discharge_kw = total_kwh_discharged / active_support_hours if active_support_hours > 0 else 0.0

            depleted = False
            depletion_time = None
            if active_intervals < len(ov_indices) and remaining_energy <= 0.05:
                depleted = True
                empty_idx = ov_indices[min(active_intervals, len(ov_indices) - 1)]
                depletion_time = fut_df.loc[empty_idx, "timestamp"].strftime("%H:%M")

            if depleted:
                ai_battery_hours = float(active_support_hours)
                ai_usage_summary = f"Depletes after {ai_battery_hours:.1f}h (at {depletion_time}) under AI predicted {avg_discharge_kw:.1f} kW peak draw"
                ai_usage_profile_detail = f"Depleted at {depletion_time} ({avg_discharge_kw:.1f} kW avg draw) — Demand Response takes over remaining overload"
            else:
                if avg_discharge_kw > 0:
                    ai_battery_hours = float(battery_storage_kwh / avg_discharge_kw)
                else:
                    ai_battery_hours = float(battery_storage_kwh / max(1.0, battery_capacity))

                remaining_kwh = max(0.0, battery_storage_kwh - total_kwh_discharged)
                ai_usage_summary = f"Covers full {total_peak_hours:.1f}h peak at {avg_discharge_kw:.1f} kW draw ({remaining_kwh:.0f} kWh reserve remaining)"
                ai_usage_profile_detail = f"Sustains full {total_peak_hours:.1f}h peak overload with {remaining_kwh:.0f} kWh buffer"
        else:
            pos_net = fut_df["net_load_kw"].clip(lower=0)
            e_sim = float(battery_storage_kwh)
            intervals_survived = 0
            for p in pos_net:
                act_p = min(float(p), float(battery_capacity), e_sim / dt)
                e_sim -= act_p * dt
                if act_p > 0:
                    intervals_survived += 1
                if e_sim <= 0.05:
                    break

            ai_battery_hours = float(intervals_survived * dt)
            avg_net_kw = float(pos_net.mean()) if len(pos_net) > 0 else 0.0
            ai_usage_summary = f"Sustains feeder for {ai_battery_hours:.1f}h under AI predicted {avg_net_kw:.1f} kW net load"
            ai_usage_profile_detail = f"Feeder backup: {ai_battery_hours:.1f} Hours at predicted consumer load profile"

        # Key Metrics
        peak_baseline = float(fut_df["net_load_kw"].max())
        peak_mitigated = float(fut_df["mitigated_net_load_kw"].max())
        peak_reduction_kw = max(0.0, peak_baseline - peak_mitigated)

        # Financial cost savings
        total_energy_shifted_kwh = float((fut_df["battery_dispatch_kw"] + fut_df["demand_response_kw"]).sum() * dt)
        cost_savings_inr = float((peak_reduction_kw * 450.0) + (total_energy_shifted_kwh * 7.0))

        # Peak dispatch values
        max_batt = float(fut_df["battery_dispatch_kw"].max())
        max_dr = float(fut_df["demand_response_kw"].max())

        # Electrical currents & appliance conversions
        current_3phase_amps = float((max_dr * 1000) / (np.sqrt(3) * 415 * 0.9)) if max_dr > 0 else 0.0
        current_1phase_amps = float((max_dr * 1000) / (230 * 0.9)) if max_dr > 0 else 0.0
        geysers_count = int(np.ceil(max_dr / 3.0)) if max_dr > 0 else 0
        acs_count = int(np.ceil(max_dr / 2.0)) if max_dr > 0 else 0
        evs_count = int(np.ceil(max_dr / 3.3)) if max_dr > 0 else 0

        return GridSimulationResult(
            fut_df=fut_df,
            transformer_limit=transformer_limit,
            battery_capacity=battery_capacity,
            battery_storage_kwh=battery_storage_kwh,
            has_overload=has_overload,
            max_overload=max_overload,
            overload_hour=overload_hour,
            peak_reduction_kw=peak_reduction_kw,
            cost_savings_inr=cost_savings_inr,
            active_dispatch_hours=active_dispatch_hours,
            total_batt_dispatched_kwh=total_batt_dispatched_kwh,
            ai_battery_hours=ai_battery_hours,
            ai_usage_summary=ai_usage_summary,
            ai_usage_profile_detail=ai_usage_profile_detail,
            max_batt=max_batt,
            max_dr=max_dr,
            current_3phase_amps=current_3phase_amps,
            current_1phase_amps=current_1phase_amps,
            geysers_count=geysers_count,
            acs_count=acs_count,
            evs_count=evs_count,
            missing_data_pct=missing_data_pct,
        )
