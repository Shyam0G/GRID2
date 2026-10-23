import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Tuple
from controllers.grid_controller import GridSimulationResult


class DashboardView:
    """Streamlit View layer presenting the visual dashboard components."""

    @staticmethod
    def render_sidebar(
        cities: Dict[str, Dict[str, float]],
        ai_metrics: Dict[str, float],
    ) -> Tuple[str, str, float, float, float, str, int, Any]:
        """Renders the left sidebar controls and returns user parameters and live weather."""
        st.sidebar.markdown("### 🌐 Live Weather Forecast (Open-Meteo)")
        selected_city = st.sidebar.selectbox("Select Location", list(cities.keys()), index=0)
        city_coords = cities[selected_city]

        from models.weather_model import WeatherService
        live_w_df = WeatherService.fetch_live_weather(city_coords["lat"], city_coords["lon"])

        weather_mode = st.sidebar.radio(
            "Forecast Data Source",
            ["🌐 Live API (AI Prediction)", "🧪 Simulation Scenarios (AI-Driven)"],
            index=0,
        )

        # AI Machine Learning Engine Badge
        st.sidebar.markdown(
            f"""
            <div style="background: #11202e; border: 1px solid #238636; border-radius: 6px; padding: 10px; font-size: 0.8rem; margin-bottom: 12px;">
                <b style="color: #2ecc71;">🤖 ACTIVE AI FORECAST ENGINE</b><br/>
                • <b>Model:</b> Gradient Boosting Regressors<br/>
                • <b>Load Model R² Score:</b> {ai_metrics['load_r2'] * 100:.1f}%<br/>
                • <b>Solar Model R² Score:</b> {ai_metrics['solar_r2'] * 100:.1f}%<br/>
                • <b>Inference:</b> Pure Machine Learning (No Math Multipliers)
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live Weather Status Badge
        if weather_mode == "🌐 Live API (AI Prediction)" and live_w_df is not None:
            curr_temp = live_w_df["temperature"].iloc[0]
            curr_cloud = live_w_df["cloud_cover"].iloc[0]
            curr_solar = live_w_df["solar_irradiance"].iloc[0]
            st.sidebar.markdown(
                f"""
                <div style="background: #11202e; border: 1px solid #1f6feb; border-radius: 6px; padding: 10px; font-size: 0.8rem; margin-bottom: 12px;">
                    <b style="color: #58a6ff;">● LIVE OPEN-METEO ACTIVE</b><br/>
                    <b>Location:</b> {selected_city}<br/>
                    <b>Temp:</b> {curr_temp:.1f}°C | <b>Clouds:</b> {curr_cloud:.0f}%<br/>
                    <b>Solar Rad:</b> {curr_solar:.0f} W/m²
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif weather_mode == "🌐 Live API (AI Prediction)":
            st.sidebar.warning("⚠️ Live API temporarily unreachable. Falling back to cached telemetry.")
            if st.sidebar.button("🔄 Reconnect Live Weather"):
                from models.weather_model import WeatherService
                WeatherService.clear_cache()
                st.rerun()

        st.sidebar.markdown("### ⚙️ Simulation Controls")
        default_limit = 150 if weather_mode == "🌐 Live API (AI Prediction)" else 115
        default_batt = 40 if weather_mode == "🌐 Live API (AI Prediction)" else 25

        transformer_limit = st.sidebar.slider(
            "Transformer Safe Capacity (kW)", min_value=80, max_value=250, value=default_limit, step=5
        )
        battery_capacity = st.sidebar.slider(
            "Available Battery Storage (kW)", min_value=0, max_value=80, value=default_batt, step=5
        )

        if weather_mode == "🧪 Simulation Scenarios (AI-Driven)":
            weather_cond = st.sidebar.selectbox(
                "Scenario Preset",
                ["Normal Day", "Cloudy Afternoon (Solar Drop)", "Sudden Evening Heatwave"],
            )
        else:
            weather_cond = "Live API"

        missing_data_pct = st.sidebar.slider(
            "Simulate Noisy/Missing Meter Data (%)", min_value=0, max_value=30, value=15, step=5
        )

        return (
            selected_city,
            weather_mode,
            float(transformer_limit),
            float(battery_capacity),
            weather_cond,
            int(missing_data_pct),
            live_w_df,
        )

    @staticmethod
    def render_header_and_banner(result: GridSimulationResult):
        """Displays application title and status/overload alert banner."""
        st.title("⚡ AI Grid Load & DR Platform")

        if result.has_overload:
            st.error(
                f"🚨 **CRITICAL ALERT: Transformer Overload Risk Detected!** "
                f"Peak excess of **{result.max_overload:.1f} kW** expected at **{result.overload_hour}**. Mitigation active."
            )
        else:
            st.success("✅ **GRID STATUS NORMAL:** Net load is within safe transformer limits for the next 24 hours.")

    @staticmethod
    def render_kpi_cards(result: GridSimulationResult):
        """Renders top 4 summary metric KPI cards."""
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">AI PREDICTED PEAK LOAD</div>
                    <div class="kpi-value">{result.fut_df['pred_load_kw'].max():.1f} kW</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">AI PREDICTED MAX SOLAR</div>
                    <div class="kpi-value">{result.fut_df['pred_solar_kw'].max():.1f} kW</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">PEAK SHAVED</div>
                    <div class="kpi-value">{result.peak_reduction_kw:.1f} kW</div>
                    <div class="kpi-delta delta-red">↓ -{result.max_overload:.1f} kW Overload</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">ESTIMATED COST SAVINGS</div>
                    <div class="kpi-value">₹ {result.cost_savings_inr:,.0f}</div>
                    <div class="kpi-delta delta-green">↑ Net Savings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

    @staticmethod
    def render_forecast_chart(result: GridSimulationResult, selected_city: str, weather_mode: str):
        """Renders the interactive Plotly 24h lookahead forecast curves."""
        chart_title = (
            f"📈 24-Hour Ahead AI Forecast — {selected_city} (Live Weather)"
            if weather_mode == "🌐 Live API (AI Prediction)"
            else "📈 24-Hour Ahead Feeder & Solar Forecast"
        )
        st.subheader(chart_title)
        fig = go.Figure()

        fut_df = result.fut_df

        # Predicted Gross Demand, Solar, and Net Load
        fig.add_trace(
            go.Scatter(
                x=fut_df["timestamp"],
                y=fut_df["pred_load_kw"],
                mode="lines",
                name="AI Predicted Demand (kW)",
                line=dict(color="#00d4ff", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fut_df["timestamp"],
                y=fut_df["pred_solar_kw"],
                mode="lines",
                name="AI Predicted Solar (kW)",
                line=dict(color="#f1c40f", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fut_df["timestamp"],
                y=fut_df["net_load_kw"],
                mode="lines",
                name="Net Load (Grid Import) (kW)",
                line=dict(color="#9b59b6", width=2.5),
            )
        )

        if result.has_overload:
            fig.add_trace(
                go.Scatter(
                    x=fut_df["timestamp"],
                    y=fut_df["mitigated_net_load_kw"],
                    mode="lines",
                    name="Mitigated Net Load (Safe)",
                    line=dict(color="#2ecc71", width=3),
                )
            )

        # Transformer Safe Threshold Line
        fig.add_hline(
            y=result.transformer_limit,
            line_dash="dash",
            line_color="#ff5555",
            annotation_text=f"Transformer Safe Limit ({result.transformer_limit:.0f} kW)",
            annotation_position="top left",
        )

        fig.update_layout(
            plot_bgcolor="#161b22",
            paper_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            height=480,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Power (kW)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_automated_mitigation_panel(result: GridSimulationResult):
        """Renders the right column mitigation actions and AI dynamic runtime."""
        st.markdown(
            f"""
            <div class="action-panel">
                <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 8px;">🛡️ Automated Mitigation</div>
                <div style="font-size: 0.95rem; margin-bottom: 4px;"><b>Transformer Rating:</b> <span style="background-color: #21262d; padding: 2px 6px; border-radius: 4px;">{result.transformer_limit:.0f} kW</span></div>
                <div style="font-size: 0.95rem; margin-bottom: 4px;"><b>Battery Reserve:</b> <span style="background-color: #21262d; padding: 2px 6px; border-radius: 4px;">{result.battery_capacity:.0f} kW</span></div>
                <div style="font-size: 0.95rem; margin-bottom: 2px;"><b>AI Predicted Runtime:</b> <span style="background-color: #21262d; padding: 2px 6px; border-radius: 4px; color: #2ecc71; font-weight: 700;">⏱️ {result.ai_battery_hours:.1f} Hours</span></div>
                <div style="font-size: 0.78rem; color: #8b949e; line-height: 1.25; margin-bottom: 14px;">🤖 <i>{result.ai_usage_summary}</i></div>
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 12px;">Real-time Actions:</div>
            """,
            unsafe_allow_html=True,
        )

        if result.has_overload:
            # Active Battery Dispatch Card
            if result.max_batt > 0:
                st.markdown(
                    f"""
                    <div class="batt-card">
                        <div style="font-weight: 700; font-size: 0.95rem;">🔋 Battery Dispatch: Discharging</div>
                        <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px;">{result.max_batt:.1f} kW peak</div>
                        <div style="font-size: 0.82rem; margin-top: 6px; opacity: 0.95;">
                            ⏱️ <b>AI Predicted Longevity:</b> {result.ai_battery_hours:.1f} Hours<br/>
                            📊 <b>AI Usage Profile:</b> {result.ai_usage_profile_detail}<br/>
                            ⚡ <b>Energy Dispatched:</b> {result.total_batt_dispatched_kwh:.1f} kWh
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Demand Response Card with Currents & Appliance reduction advice
            if result.max_dr > 0:
                st.markdown(
                    f"""
                    <div class="dr-card">
                        <div style="font-weight: 700; font-size: 0.95rem;">📡 Demand Response:</div>
                        <div style="font-size: 1.15rem; font-weight: 700; margin-top: 4px;">Curtailed/Shifted {result.max_dr:.1f} kW</div>
                        <div style="font-size: 0.82rem; margin-top: 4px; opacity: 0.9;">
                            Feeder Current to Cut: <b>{result.current_3phase_amps:.1f} A</b> (3-Phase)<br/>
                            Total Consumer Draw: <b>{result.current_1phase_amps:.1f} A</b> (@ 230V)
                        </div>
                    </div>

                    <div class="advisory-box">
                        <b>📢 Consumer Load Cut Advisory:</b><br/>
                        To prevent feeder circuit breaker trip, the consumers connected to this transformer must turn off <b>{result.current_1phase_amps:.1f} A</b> total:<br/>
                        • Pause <b>{result.evs_count}</b> EV Fast Chargers (~15 A each), OR<br/>
                        • Turn off <b>{result.acs_count}</b> Air Conditioners (~9 A each), OR<br/>
                        • Switch off <b>{result.geysers_count}</b> Water Geysers (~13 A each).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success("📡 Demand Response: Not needed (Battery covered 100%)")
        else:
            st.success("Grid is stable. No active curtailment required.")

        st.markdown(
            f"""
                <hr style="border-color: #30363d; margin-top: 15px; margin-bottom: 10px;"/>
                <div style="font-size: 0.78rem; color: #8b949e;">
                    <b>Data Pipeline Integrity:</b> {100 - result.missing_data_pct}% clean raw data.<br/>
                    Missing intervals auto-interpolated without disruption.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
