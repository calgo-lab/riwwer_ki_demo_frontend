import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import plotly.graph_objs as go
import altair as alt
from utils.config import (
    read_data,
    TARGET_COLUMN,
    MAP_DATA,
    COORDINATES_DICT,
    SENSOR_GROUPS,
    calculate_target_column_bounds,
    RAINFALL_COLUMN,
    RAINFALL_FORECAST_COLUMN,
    LOCAL_PREDICTIONS_PATH,
    PREDICTIONS_TIME_COLUMN,
    LOCAL_LSTM_PRED_COLUMN,
    LOCAL_TRANSFORMER_PRED_COLUMN,
    GLOBAL_PREDICTIONS_PATH,
    GLOBAL_TFT_PRED_COLUMN,
    GLOBAL_LSTM_PRED_COLUMN,
    OVERFLOW_CLS_PRED_FILE_PATH
)
from utils.dynamic_map_data import build_dynamic_map_data_from_row, load_map_data
from components import TimeSliderLive, SimulationChart, RainfallBarChart, OverflowRiskometer

st.set_page_config(
    page_title="RIWWER KI Demo",
    page_icon="⚡",
    layout="wide",
)

st.title("RIWWER KI Demo")

# Styles for bordered panels with a blue title bar
st.markdown(
    """
    <style>
    .panel-header {
        background: #c2cbfc;
        border-left: 4px solid #1f6feb;
        color: #0b63ce;
        padding: 0.4rem 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]
map_data = load_map_data() # Why not just use vierlinden_data directly?

# Load local model predictions (one-step ahead)
@st.cache_data(ttl=900)
def _load_local_preds(path: str, time_col: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
    except Exception:
        return None

local_preds = _load_local_preds(LOCAL_PREDICTIONS_PATH, PREDICTIONS_TIME_COLUMN)

# Load global multi-step predictions (arrays of length 12 per timestamp)
@st.cache_data(ttl=900)
def _load_global_preds(path: str, time_col: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
        import ast
        for col in [GLOBAL_TFT_PRED_COLUMN, GLOBAL_LSTM_PRED_COLUMN]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        return df
    except Exception:
        return None

global_preds = _load_global_preds(GLOBAL_PREDICTIONS_PATH, PREDICTIONS_TIME_COLUMN)

# Load overflow probability for riskometer predictions
@st.cache_data(ttl=900)
def _load_overflow_cls_predictions() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(OVERFLOW_CLS_PRED_FILE_PATH, parse_dates=[PREDICTIONS_TIME_COLUMN],
                                                    index_col=PREDICTIONS_TIME_COLUMN)
        return df
    except Exception:
        return None
overflow_cls_predictions = _load_overflow_cls_predictions()

# Calculate fixed y-axis bounds for consistent chart scaling
y_axis_bounds = calculate_target_column_bounds(vierlinden_data)

# # Show data overview
# with st.expander("📊 Data Overview", expanded=False):
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Records", len(vierlinden_data))
#     with col2:
#         st.metric(
#             "Date Range", f"{vierlinden_data.index.min().strftime('%Y-%m-%d')}"
#         )
#         st.caption(f"to {vierlinden_data.index.max().strftime('%Y-%m-%d')}")
#     with col3:
#         st.metric("Target Column", TARGET_COLUMN)
#     with col4:
#         st.metric("Y-Axis Range", f"[{y_axis_bounds[0]:.1f}, {y_axis_bounds[1]:.1f}]")
#         st.caption("Fixed bounds (±0.5 rounded)")

# Configuration options (now expandable)
with st.expander("⚙️ Dashboard Configuration", expanded=False):
    # Model scope selector
    model_scope = st.radio(
        "Model scope",
        options=["Standard operation", "Full network outage"],
        index=0,
        horizontal=True,
        key="model_scope_selector",
        help="Standard operation: Uses all available sensor information within the network. Full network outage: Treats all external sensor information as inactive and use only local measurements for the forecasts."
    )
    is_local_mode = (model_scope == "Full network outage")
    is_global_mode = (model_scope == "Standard operation")

    # Local model selector (only visible in Local mode)
    local_model_choice = None
    global_model_choice = None
    if is_local_mode:
        local_model_choice = st.radio(
            "Local model",
            options=["LSTM", "Transformer"],
            index=0,
            horizontal=True,
            key="local_model_selector",
            help="Choose which local model's predictions to visualize."
        )
    elif is_global_mode:
        global_model_choice = st.radio(
            "Global model",
            options=["TFT", "LSTM"],
            index=0,
            horizontal=True,
            key="global_model_selector",
            help="Choose which global model's predictions to visualize (12-step ahead)."
        )

    # Chart renderer selection
    renderer_choice = st.radio(
        "Chart renderer",
        options=["Bokeh", "Matplotlib"],
        index=0,
        horizontal=True,
        key="chart_renderer_selector",
        help="Bokeh for smooth interactive, Matplotlib for lightweight static."
    )

# Initialize components using the smooth TimeSliderLive approach
time_slider = TimeSliderLive(vierlinden_data, session_key="pydeck_main")
simulation_chart = SimulationChart(
    key="pydeck_simulation_chart",
    interactive=True,
    renderer=("bokeh" if renderer_choice == "Bokeh" else "matplotlib"),
)
rainfall_chart = RainfallBarChart(key="pydeck_rainfall_chart")

# Pre-calculate map center based on bounding box (not just coordinate average)
# Get min/max bounds of all locations
lats = [coord[1] for coord in COORDINATES_DICT.values()]
lons = [coord[0] for coord in COORDINATES_DICT.values()]

min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)

# Calculate true geographic center of bounding box
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# Optional: Calculate optimal zoom level based on bounding box size
lat_range = max_lat - min_lat
lon_range = max_lon - min_lon
max_range = max(lat_range, lon_range)

# ========== MARKER SIZE CONFIGURATION ==========
# Adjust these values to fine-tune marker appearance
BASE_MARKER_SIZE = 80        # Base size for all markers (consistent across locations)
RING_SIZE_MULTIPLIER = 1.7    # How much larger the outer ring should be
ICON_SIZE_MULTIPLIER = 0.25   # Size of the center white dot relative to base
# ===============================================

def get_marker_color_and_size(sensor_statuses):
    """Determine marker color based on sensor status - SIZE IS NOW CONSISTENT"""
    if not sensor_statuses:
        return [128, 128, 128, 180]  # Gray for no sensors

    active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
    total_sensors = len(sensor_statuses)

    if total_sensors == 0:
        return [128, 128, 128, 180]  # Gray
    elif active_sensors == total_sensors:
        return [0, 255, 0, 200]  # Green - all active
    elif active_sensors > 0:
        return [255, 165, 0, 200]  # Orange - partial
    else:
        return [255, 0, 0, 200]  # Red - all inactive

def prepare_map_data(data_row, timestamp, model_scope: str):
    """Prepare enhanced marker data for Pydeck visualization"""
    # Build dynamic map data using current row
    dynamic_map_data = build_dynamic_map_data_from_row(map_data, data_row, timestamp)

    # Prepare data for multiple layer types
    marker_base_points = []  # Base circles
    marker_ring_points = []  # Outer rings for emphasis
    marker_icons = []        # Icon-like center points
    label_points = []        # Text labels
    target_highlight_points = []  # Subtle highlight for Kläranlage

    for _, row in dynamic_map_data.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        location_name = row["info"]
        sensor_statuses = row["sensor_statuses"]
        is_target_location = (location_name == "Kläranlage")

        # In Local mode, force all sensors to inactive for each location
        if model_scope == "Full network outage":
            sensor_list = SENSOR_GROUPS.get(location_name, [])
            sensor_statuses = [
                {
                    "Sensor": s,
                    "Status": "inactive",
                    "Value": None,
                    "LastValidDataTime": None,
                    "TimeSinceLastValidData": None,
                }
                for s in sensor_list
            ]

        # Get color based on sensor status (size is now consistent)
        color = get_marker_color_and_size(sensor_statuses)

        # Calculate status metrics
        active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
        total_sensors = len(sensor_statuses)
        activity_percentage = 100 * (active_sensors / max(total_sensors, 1))

        # Use consistent sizes for all markers
        base_size = BASE_MARKER_SIZE
        ring_size = BASE_MARKER_SIZE * RING_SIZE_MULTIPLIER
        icon_size = BASE_MARKER_SIZE * ICON_SIZE_MULTIPLIER

        # 1. Base marker (filled circle) - CONSISTENT SIZE
        marker_base_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': color,
            'size': base_size,  # Same for all locations
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 0,
            'note': 'Sewage Treatment Facility<br>Location of filling level forecasts' if is_target_location else ''
        })

        # 2. Outer ring for emphasis - CONSISTENT SIZE
        ring_color = color[:3] + [80]  # Same color but more transparent
        marker_ring_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': ring_color,
            'size': ring_size,  # Consistent ring size
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 5
        })

        # 3. Icon center - CONSISTENT SIZE
        marker_icons.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': [156, 39, 176, 255] if is_target_location else [255, 255, 255, 255],  # Purple for target
            'size': icon_size,  # Consistent icon size
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 10
        })

        # 3b. Dedicated subtle highlight ring for Kläranlage (thin azure outline)
        if is_target_location:
            target_highlight_points.append({
                'lon': lon,
                'lat': lat,
                'location': location_name,
                # Larger than the standard ring to appear as a distinct outer ring
                'size': base_size * (RING_SIZE_MULTIPLIER + 0.2),
                'elevation': 8
            })

        # 4. Labels with better positioning
        # Determine label color based on activity
        label_color = [255, 255, 255, 255] if activity_percentage > 0.5 else [0, 0, 0, 255]
        label_bg_color = [0, 0, 0, 180] if activity_percentage > 0.5 else [255, 255, 255, 200]

        label_points.append({
            'lon': lon,
            'lat': lat - 0.0008,  # Position below marker
            'text': location_name,
            'size': 11,
            'color': label_color,
            'background_color': label_bg_color,
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors
        })

    return (
        pd.DataFrame(marker_base_points),
        pd.DataFrame(marker_ring_points),
        pd.DataFrame(marker_icons),
        pd.DataFrame(label_points),
        pd.DataFrame(target_highlight_points)
    )

def smooth_content_renderer(idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int):
    """Content renderer for smooth updates using the TimeSliderLive pattern"""

    # Current data display at the top
    st.subheader("📊 Current Data Point")

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Date", timestamp.strftime("%Y-%m-%d"))
    with col2:
        st.metric("Time", f"{timestamp.strftime('%H:%M:%S')}")
    with col3:
        current_value = data_row[TARGET_COLUMN]
        st.metric("Filling level - Overflow Basin", f"{current_value:.2f}")
    with col4:
        # Rainfall value (if available)
        try:
            if RAINFALL_COLUMN in data_row.index and pd.notna(data_row[RAINFALL_COLUMN]):
                st.metric("Rainfall (mm)", f"{float(data_row[RAINFALL_COLUMN]):.2f}")
            else:
                st.metric("Rainfall (mm)", "-")
        except Exception:
            st.metric("Rainfall (mm)", "-")

    # Main content area: panels (equal widths)
    left_panel_col, right_panel_col = st.columns([1,2], gap="small")

    # Left Panel: Real-time Sensor Map (full width) with Overview below
    with left_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>📍 Sensor Map & 📊 Overview</div>", unsafe_allow_html=True)

            # Prepare enhanced map data with multiple layers
            marker_base_data, marker_ring_data, marker_icon_data, label_data, target_highlight_data = prepare_map_data(data_row, timestamp, model_scope)

            # Create multiple layers for marker-like appearance
            layers = []

            # 1. Outer ring layer (rendered first, behind everything)
            ring_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_ring_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                pickable=False,
                stroked=True,
                stroke_width_min_pixels=1,
                stroke_width_max_pixels=2,
                get_line_color=[255, 255, 255, 160],
                elevation_scale=1,
                elevation_range=[0, 20],
                get_elevation='elevation'
            )
            layers.append(ring_layer)

            # 1b. Subtle highlight ring for Kläranlage only (bold purple outline)
            if target_highlight_data is not None and not target_highlight_data.empty:
                target_layer = pdk.Layer(
                    'ScatterplotLayer',
                    target_highlight_data,
                    get_position='[lon, lat]',
                    get_color=[156, 39, 176, 0],  # no fill, purple hue
                    get_radius='size',
                    pickable=False,
                    filled=False,               # render as ring only
                    stroked=True,
                    line_width_min_pixels=5,  # bolder outline
                    line_width_max_pixels=5,
                    get_line_color=[156, 39, 176, 245],
                    elevation_scale=2,
                    elevation_range=[0, 30],
                    get_elevation='elevation'
                )
                layers.append(target_layer)

            # 2. Base marker layer (main colored circle)
            marker_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_base_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                pickable=True,
                auto_highlight=True,
                stroked=True,
                stroke_width_min_pixels=2,
                stroke_width_max_pixels=3,
                get_line_color=[255, 255, 255, 200],
                elevation_scale=2,
                elevation_range=[0, 30],
                get_elevation='elevation'
            )
            layers.append(marker_layer)

            # 3. Icon center layer (white dot to simulate RSS icon)
            icon_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_icon_data,
                get_position='[lon, lat]',
                get_color=[255, 255, 255, 255],
                get_radius='size',
                pickable=False,
                elevation_scale=3,
                elevation_range=[0, 40],
                get_elevation='elevation'
            )
            layers.append(icon_layer)

            # 4. Text labels with background effect
            if label_data is not None and not label_data.empty:
                # Background layer for labels (slightly larger, darker)
                label_bg_layer = pdk.Layer(
                    'TextLayer',
                    label_data,
                    get_position='[lon, lat]',
                    get_text='text',
                    get_color=[128, 128, 128, 128],  # Black text
                    get_size=13,  # Slightly larger for background effect
                    get_alignment_baseline="'middle'",
                    get_text_anchor="'middle'",
                    font_weight='bold'
                )
                layers.append(label_bg_layer)

                # Foreground text layer
                text_layer = pdk.Layer(
                    'TextLayer',
                    label_data,
                    get_position='[lon, lat]',
                    get_text='text',
                    get_color=[255, 255, 255, 255],  # White text
                    get_size=11,
                    get_alignment_baseline="'middle'",
                    get_text_anchor="'middle'",
                    font_weight='bold'
                )
                layers.append(text_layer)

                # Create deck with stable view state (no flicker!)
                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=13.25,
                    pitch=0,  # Slight 3D angle to show elevation
                    bearing=0
                )

                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        'html': '''
                            <div style="background: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px;">
                                <b>📍 {location}</b><br/>
                                <span style="color: #4CAF50;">●</span> Active: {active_sensors}<br/>
                                <span style="color: #FF5722;">●</span> Inactive: {inactive_sensors}<br/>
                                <i>{note}</i>
                            </div>
                        ''',
                        'style': {
                            'backgroundColor': 'transparent',
                            'border': 'none'
                        }
                    }
                )

            # This is the key - using a unique key based on iteration prevents flickering
            st.pydeck_chart(
                deck,
                use_container_width=True,
                key=f"enhanced_pydeck_map_{iteration}",
                height=420,
            )

            # Current Overview metrics stacked beneath the map
            st.markdown("<div style='max-height:180px; overflow:auto;'>", unsafe_allow_html=True)
            # Get the map data for overview calculations
            if is_local_mode:
                st.warning(
                    "Local mode is active. All locations are treated as inactive and sensor measurements are not considered.",
                    icon="⚠️",
                )
            marker_base_data, _, _, _, _ = prepare_map_data(data_row, timestamp, model_scope)

            # Calculate overview statistics
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
                # Calculate overall network health
                if total_locations > 0:
                    network_health = ((active_locations * 100)) / total_locations
                    st.metric("🎯 Network Health", f"{network_health:.1f}%")
                    if is_local_mode:
                        st.caption("Network health reflects forced inactivity in Local mode.")
            with col2:
                st.markdown("🟣 **Sewage Treatment Facility**<br/>Overflow basins and Rainfall sensors are located here.<br/>*Location of filling level forecasts*", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Right Panel: Simulation Chart
    with right_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>📈 Filling Level History and Model Forecasts</div>", unsafe_allow_html=True)
            # Compute local one-step-ahead forecast value if in Local mode
            forecast_value = None
            forecast_series = None
            if is_local_mode and local_preds is not None:
                try:
                    # Find t+1 timestamp based on main data index
                    if idx + 1 < len(vierlinden_data.index):
                        t_plus_one = vierlinden_data.index[idx + 1]
                        if t_plus_one in local_preds.index:
                            val = local_preds.loc[t_plus_one]
                            # Use selected model column from config
                            if local_model_choice == "LSTM" and LOCAL_LSTM_PRED_COLUMN in val.index and pd.notna(val[LOCAL_LSTM_PRED_COLUMN]):
                                forecast_value = float(val[LOCAL_LSTM_PRED_COLUMN])
                            elif local_model_choice == "Transformer" and LOCAL_TRANSFORMER_PRED_COLUMN in val.index and pd.notna(val[LOCAL_TRANSFORMER_PRED_COLUMN]):
                                forecast_value = float(val[LOCAL_TRANSFORMER_PRED_COLUMN])
                except Exception:
                    forecast_value = None
            # Compute 12-step forecast series if in Global mode
            if is_global_mode and global_preds is not None:
                try:
                    # The row at t contains [t+1..t+12]
                    t_key = vierlinden_data.index[idx]
                    if t_key in global_preds.index:
                        row = global_preds.loc[t_key]
                        col = GLOBAL_TFT_PRED_COLUMN if global_model_choice == "TFT" else GLOBAL_LSTM_PRED_COLUMN
                        if col in row.index:
                            val = row[col]
                            if isinstance(val, (list, tuple)):
                                arr = val
                            elif isinstance(val, str) and val.startswith("["):
                                import ast
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
                iteration=iteration,
                show_checkbox=False,  # Disable internal checkbox to avoid conflicts
                y_axis_bounds=y_axis_bounds,  # Fixed y-axis bounds for consistent scaling
                height=635,
                forecast_value=forecast_value,  # Local one-step ahead
                forecast_series=forecast_series,   # Global multi-step ahead [t+1..t+12]
                is_local_mode=is_local_mode
            )

    with left_panel_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>Overflow Risk in the coming 2 hours</div>", unsafe_allow_html=True)
            overflow_riskometer = OverflowRiskometer(overflow_cls_predictions, timestamp, is_local_mode=is_local_mode)
            overflow_riskometer.render()
            st.markdown("</div>", unsafe_allow_html=True)

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
                is_playing=st.session_state.get("pydeck_main_is_playing", False),
            )
            st.markdown("</div>", unsafe_allow_html=True)

# Run the smooth live dashboard using TimeSliderLive
time_slider.run_live_dashboard(
    content_renderer=smooth_content_renderer,
    updates_per_second=4.0,  # Smooth update rate
    show_controls=True,
    max_iterations=2000
)
