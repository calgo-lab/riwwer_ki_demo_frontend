"""Content renderer for RIWWER ML Demo."""
import ast
import pandas as pd
import streamlit as st

from components import SimulationChart, RainfallBarChart, OverflowRiskometer
from utils.config import (
    TARGET_COLUMN,
    RAINFALL_COLUMN,
    RAINFALL_FORECAST_COLUMN,
    LOCAL_LSTM_PRED_COLUMN,
    LOCAL_TRANSFORMER_PRED_COLUMN,
    GLOBAL_TFT_PRED_COLUMN,
    GLOBAL_LSTM_PRED_COLUMN,
)
from .map import prepare_map_data, render_map


# Module-level chart instances (initialized once)
simulation_chart = SimulationChart(key="pydeck_simulation_chart", interactive=True)
rainfall_chart = RainfallBarChart(key="pydeck_rainfall_chart")


def render_current_metrics(timestamp: pd.Timestamp, data_row: pd.Series) -> None:
    """Render current data point metrics.
    
    Args:
        timestamp: Current timestamp.
        data_row: Current data row.
    """
    with st.container(border=True):
        st.markdown("<div class='panel-header'>📊 Current Data Point</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Date", timestamp.strftime("%Y-%m-%d"))
        with col2:
            st.metric("Time", f"{timestamp.strftime('%H:%M:%S')}")
        with col3:
            current_value = data_row[TARGET_COLUMN]
            st.metric("Filling level - Overflow Basin", f"{current_value:.2f}")
        with col4:
            try:
                if RAINFALL_COLUMN in data_row.index and pd.notna(data_row[RAINFALL_COLUMN]):
                    st.metric("Rainfall (mm)", f"{float(data_row[RAINFALL_COLUMN]):.2f}")
                else:
                    st.metric("Rainfall (mm)", "-")
            except Exception:
                st.metric("Rainfall (mm)", "-")


def render_network_overview(
    marker_base_data: pd.DataFrame,
    is_local_mode: bool,
) -> None:
    """Render network overview metrics.
    
    Args:
        marker_base_data: Marker base DataFrame.
        is_local_mode: Whether in local mode.
    """
    if is_local_mode:
        st.warning(
            "Local mode is active. All locations are treated as inactive and sensor measurements are not considered.",
            icon="⚠️",
        )

    total_locations = len(marker_base_data)
    active_locations = len(marker_base_data[marker_base_data['activity_percentage'] == 100])
    inactive_locations = len(marker_base_data[marker_base_data['activity_percentage'] == 0])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Total Locations", total_locations)
    with col2:
        st.metric("🟢 Fully Active", active_locations)
    with col3:
        st.metric("🔴 Inactive", inactive_locations)

    col1, col2 = st.columns([1, 2])
    with col1:
        if total_locations > 0:
            network_health = ((active_locations * 100)) / total_locations
            st.metric("🎯 Network Health", f"{network_health:.1f}%")
            if is_local_mode:
                st.caption("Network health reflects forced inactivity in Local mode.")
    with col2:
        st.markdown(
            "🟣 **Sewage Treatment Facility**<br/>Overflow basins and Rainfall sensors are located here.<br/>*Location of filling level forecasts*",
            unsafe_allow_html=True
        )


def smooth_content_renderer(
    idx: int,
    timestamp: pd.Timestamp,
    data_row: pd.Series,
    iteration: int,
    vierlinden_data: pd.DataFrame,
    map_data: pd.DataFrame,
    local_preds: pd.DataFrame | None,
    global_preds: pd.DataFrame | None,
    overflow_cls_predictions: pd.DataFrame | None,
    y_axis_bounds: tuple,
    center_lat: float,
    center_lon: float,
) -> None:
    """Content renderer for smooth updates.
    
    Args:
        idx: Current index.
        timestamp: Current timestamp.
        data_row: Current data row.
        iteration: Iteration number for chart keys.
        vierlinden_data: Full data DataFrame.
        map_data: Map data DataFrame.
        local_preds: Local predictions DataFrame.
        global_preds: Global predictions DataFrame.
        overflow_cls_predictions: Overflow predictions DataFrame.
        y_axis_bounds: Y-axis bounds tuple.
        center_lat: Map center latitude.
        center_lon: Map center longitude.
    """
    # Read configuration from session_state
    model_scope = st.session_state.get("model_scope_selector", "Standard operation")
    is_local_mode = (model_scope == "Full network outage")
    is_global_mode = not is_local_mode
    local_model_choice = st.session_state.get("local_model_selector", "LSTM")
    global_model_choice = st.session_state.get("global_model_selector", "TFT")

    # Current data display
    render_current_metrics(timestamp, data_row)

    # Main content area
    left_panel_col, right_panel_col = st.columns([1, 2], gap="small")

    # Left Panel: Sensor Map & Overview
    with left_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>📍 Sensor Map & 📊 Overview</div>", unsafe_allow_html=True)

            # Prepare map data
            marker_base_data, marker_ring_data, marker_icon_data, label_data, target_highlight_data = prepare_map_data(
                map_data, data_row, timestamp, model_scope
            )

            # Render map
            render_map(
                center_lat, center_lon,
                marker_base_data, marker_ring_data, marker_icon_data,
                label_data, target_highlight_data
            )

            # Network overview
            st.markdown("<div style='max-height:180px; overflow:auto;'>", unsafe_allow_html=True)
            render_network_overview(marker_base_data, is_local_mode)
            st.markdown("</div>", unsafe_allow_html=True)

    # Right Panel: Simulation Chart
    with right_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>📈 Filling Level History and Model Forecasts</div>", unsafe_allow_html=True)

            # Compute forecast values
            forecast_value = None
            forecast_series = None
            current_value = data_row[TARGET_COLUMN]

            if is_local_mode and local_preds is not None:
                try:
                    if idx + 1 < len(vierlinden_data.index):
                        t_plus_one = vierlinden_data.index[idx + 1]
                        if t_plus_one in local_preds.index:
                            val = local_preds.loc[t_plus_one]
                            if local_model_choice == "LSTM" and LOCAL_LSTM_PRED_COLUMN in val.index and pd.notna(val[LOCAL_LSTM_PRED_COLUMN]):
                                forecast_value = float(val[LOCAL_LSTM_PRED_COLUMN])
                            elif local_model_choice == "Transformer" and LOCAL_TRANSFORMER_PRED_COLUMN in val.index and pd.notna(val[LOCAL_TRANSFORMER_PRED_COLUMN]):
                                forecast_value = float(val[LOCAL_TRANSFORMER_PRED_COLUMN])
                except Exception:
                    forecast_value = None

            if is_global_mode and global_preds is not None:
                try:
                    t_key = vierlinden_data.index[idx]
                    if t_key in global_preds.index:
                        row = global_preds.loc[t_key]
                        col = GLOBAL_TFT_PRED_COLUMN if global_model_choice == "TFT" else GLOBAL_LSTM_PRED_COLUMN
                        if col in row.index:
                            val = row[col]
                            if isinstance(val, (list, tuple)):
                                arr = val
                            elif isinstance(val, str) and val.startswith("["):
                                arr = ast.literal_eval(val)
                            else:
                                arr = None
                            if isinstance(arr, (list, tuple)) and len(arr) >= 1:
                                forecast_series = [float(x) for x in list(arr)[:12]]
                except Exception:
                    forecast_series = None

            simulation_chart.render(
                data=vierlinden_data,
                current_timestamp=timestamp,
                current_value=current_value,
                target_column=TARGET_COLUMN,
                iteration=None,
                show_checkbox=False,
                y_axis_bounds=y_axis_bounds,
                height=635,
                forecast_value=forecast_value,
                forecast_series=forecast_series,
                is_local_mode=is_local_mode
            )

    # Overflow Risk
    with left_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>Overflow Risk in the coming 2 hours</div>", unsafe_allow_html=True)
            overflow_riskometer = OverflowRiskometer(overflow_cls_predictions, timestamp, is_local_mode=is_local_mode)
            overflow_riskometer.render()
            st.markdown("</div>", unsafe_allow_html=True)

    # Rainfall Chart
    with right_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>🌧️ Rainfall 72h-History (upper) and 12h-Forecast (lower)</div>", unsafe_allow_html=True)
            rainfall_chart.render(
                data=vierlinden_data,
                current_timestamp=timestamp,
                rainfall_column=RAINFALL_COLUMN,
                rainfall_forecast_column=RAINFALL_FORECAST_COLUMN,
                default_history_hours=72,
                future_hours=12,
                height=260,
                show_controls=False,
                is_playing=st.session_state.get("is_playing", False),
            )
            st.markdown("</div>", unsafe_allow_html=True)