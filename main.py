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
    CACHE_TIME,
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
from components import SimulationChart, RainfallBarChart, OverflowRiskometer
from PIL import Image

# Min/Max Update intervals for auto-advance when playing
MIN_UPDATE_INTERVAL = 0.2 # to low update interval could cause performance issues
MAX_UPDATE_INTERVAL = 1.0

# Check Streamlit version for fragment support
STREAMLIT_VERSION = tuple(int(x) for x in st.__version__.split('.'))
if STREAMLIT_VERSION < (1, 35, 0):
    st.error(f"Streamlit {st.__version__} does not support fragments. Please upgrade to 1.35.0 or later.")
    st.stop()

icon = Image.open("figures/icon.png")

st.set_page_config(
    page_title="RIWWER ML Demo",
    page_icon=icon,
    layout="wide",
)

st.title("Forecasting Models for Urban Wastewater Management")

# =============================================================================
# STATIC UI FUNCTIONS (run once at startup)
# =============================================================================

def render_static_header():
    """Render static header elements (logos, intro, CSS). Runs once at startup."""
    
    # Logos row (uniform height, concatenated with fixed spacing)
    try:
        from pathlib import Path
        import base64

        def _b64_img(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")

        logos = [
            "figures/riwwer-logo.png",
            "figures/bht-logo.png",
            "figures/calgolab-logo.png",
            "figures/okeanos-logo.png",
            "figures/ude-logo.png",
        ]
        logo_files = [p for p in logos if os.path.exists(p)]

        if logo_files:
            logo_h = 100  # px fixed height
            gap_px = 24   # fixed horizontal spacing
            # CSS for a single-row centered flex layout with fixed gaps
            st.markdown(
                f"""
                <style>
                .logo-row {{
                    display: flex;
                    align-items: center;
                    justify-content: left;
                    gap: {gap_px}px;
                    flex-wrap: nowrap;
                    margin-bottom: 8px;
                }}
                .logo-row img {{
                    height: {logo_h}px;
                    object-fit: contain;
                    display: inline-block;
                }}
                @media (max-width: 900px) {{
                    .logo-row {{ flex-wrap: wrap; gap: 16px; }}
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Build the row of images
            imgs_html = []
            for path in logo_files:
                try:
                    b64 = _b64_img(path)
                    imgs_html.append(
                        f"<img src='data:image/png;base64,{b64}' alt='{os.path.basename(path)}' />"
                    )
                except Exception:
                    imgs_html.append(
                        f"<img src='{path}' alt='{os.path.basename(path)}' />"
                    )

            st.markdown("<div class='logo-row'>" + "".join(imgs_html) + "</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # Short intro text (full-width, styled for readability)
    # Pick text color depending on Streamlit theme (white on dark theme)
    # Theme-aware styling
    try:
        current_theme = st.context.theme.type
    except Exception:
        current_theme = 'light'
    is_dark = (current_theme == 'dark')
    _intro_text_color = "#ffffff" if is_dark else "#111111"
    st.markdown(
        f"""
        <div style="width:100%; box-sizing:border-box; padding:0 16px; font-size:19px; line-height:1.45; color:{_intro_text_color};">
        <strong>Welcome to the RIWWER ML Demo!</strong> This application showcases the machine learning (ML) models for Urban Wastewater Management
        developed by the Berliner Hochschule für Technik, Okeanos and the University of Duisburg-Essen. 
        The models are applied to historical data from the combined sewer system of Vierlinden in Duisburg (<em>Wirtschaftsbetriebe Duisburg</em>).
        We demonstrate the performance of ML models to forecast filling levels and estimate the risk of Combined Sewer Overflows in the year 2023.
        The models were trained with data from the years of 2021 and 2022. For further information consult our GitHub repository: <a href="https://github.com/calgo-lab/resilient-timeseries-evaluation">resilient-timeseries-evaluation</a><br/>
        <br/>
        Start by navigating through time using the buttons and sliders in the <strong>Time Navigation Control</strong>. Alternatively you can also search for specific rain events using the <em>"Select rainfall"</em> slider.<br/>
        <br/>
        The project was funded by the Federal Ministry of Economic Affairs and Climate Action of Germany for the RIWWER project (01MD22007H, 01MD22007C).<br/>
        <em>RIWWER: Reduction of the Impact of untreated WasteWater on the Environment in case of torrential Rain</em>
        <br/><br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Demo video & citation rendered in the static header to avoid being re-created by fragments
    try:
        video_url = "https://cloud.bht-berlin.de/public.php/dav/files/b9xt4T3SdiLBiFZ"
        # Put Demo Video and Cite Us side-by-side in columns (4:2)
        cols = st.columns([5, 4], gap="small")
        # Left: embedded video
        with cols[0]:
            with st.container(border=True):
                st.markdown("<div class='panel-header'>🎬 Demo Video</div>", unsafe_allow_html=True)
                try:
                    html5_video = (
                        f'<div style="display:flex;justify-content:center;">'
                        f'<div style="width:min(100%, 75rem);">'
                        f'<video controls preload="metadata" style="display:block;width:100%;height:auto;aspect-ratio:16/9;border-radius:8px;" src="{video_url}">'
                        f'Your browser does not support the video tag.'
                        f'</video>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(html5_video, unsafe_allow_html=True)
                except Exception:
                    st.markdown(
                        f"<div style='text-align:center; padding:12px 8px;'><a href='{video_url}' target='_blank' style='display:inline-block; padding:10px 18px; background:#1f6feb; color:#fff; border-radius:6px; text-decoration:none; font-weight:600;'>▶️ Open Video</a></div>",
                        unsafe_allow_html=True,
                    )
        # Right: citation and links with matching font size to intro
        with cols[1]:
            with st.container(border=True):
                st.markdown("<div class='panel-header'>📚 Learn More & Cite Us</div>", unsafe_allow_html=True)
                pub_html = f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'>\n<strong>Publication:</strong><br/>\n\"A Resilient Solution for Sewer Overflow Monitoring across Cloud and Edge\"\n</div>"
                st.markdown(pub_html, unsafe_allow_html=True)

                citation_text = '''@article{singh2026resilientsolutionseweroverflow,
    title={A Resilient Solution for Sewer Overflow Monitoring across Cloud and Edge}, 
    author={Vipin Singh and Tianheng Ling and Peter Ghaly and Felix Grimmeisen and Gregor Schiele and Felix Biessmann},
    year={2026},
    eprint={2605.10592},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2605.10592}, 
}'''

                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'><strong>BibTeX Citation:</strong></div>", unsafe_allow_html=True)            
                st.code(citation_text, language="bibtex")
                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'><strong>Paper URL (arXiv):</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:19px; line-height:1.45;'><a href='https://arxiv.org/abs/2605.10592' target='_blank'>https://arxiv.org/abs/2605.10592</a></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'>accepted at 35th International Joint Conference on Artificial Intelligence 2026 (IJCAI-ECAI 2026), Demonstrations Track.</div>", unsafe_allow_html=True)
    except Exception:
        # Avoid breaking the static header if any of these elements fail
        pass


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables for time navigation."""
    
    # Core time navigation state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
        
    if "current_update_time_interval" not in st.session_state:
        st.session_state.current_update_time_interval = None
    
    # Note: speed_multiplier is NOT initialized here - let the selectbox widget handle it
    # Using .get() with default in dynamic_content_fragment instead
    
    # Timing accumulators for smooth playback
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = time.time()
    
    if "accumulated_time" not in st.session_state:
        st.session_state.accumulated_time = 0.0
    
    # Skip auto-advance flag (prevents double-step on play)
    if "skip_autoadvance" not in st.session_state:
        st.session_state.skip_autoadvance = False
    
    # Rainfall slider versioning and suppression
    if "rain_slider_ver" not in st.session_state:
        st.session_state.rain_slider_ver = 0
    
    if "rain_suppress" not in st.session_state:
        st.session_state.rain_suppress = False
    
    if "rain_last_sig" not in st.session_state:
        st.session_state.rain_last_sig = ""


def _reset_timing_state():
    """Reset timing accumulators used by play/auto-advance."""
    st.session_state.last_update_time = time.time()
    st.session_state.accumulated_time = 0.0
    st.session_state.skip_autoadvance = True


def _step_backward():
    """Step one index back (bounded at 0), mirroring Back button behavior."""
    st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
    _reset_timing_state()


def _on_scope_change():
    """Callback when model scope changes - sets flag to trigger rerun after event handling."""
    st.session_state._scope_changed = True


# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]
map_data = load_map_data() # Why not just use vierlinden_data directly?

# Load local model predictions (one-step ahead)
@st.cache_data(ttl=CACHE_TIME)
def _load_local_preds(path: str, time_col: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
    except Exception:
        return None

local_preds = _load_local_preds(LOCAL_PREDICTIONS_PATH, PREDICTIONS_TIME_COLUMN)

# Load global multi-step predictions (arrays of length 12 per timestamp)
@st.cache_data(ttl=CACHE_TIME)
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
@st.cache_data(ttl=CACHE_TIME)
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

def render_dashboard_config_panel():
    with st.container(border=True):
        st.markdown("<div class='panel-header'>⚙️ Dashboard Configuration</div>", unsafe_allow_html=True)
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                Select between Standard Operation and Full Network Outage scenarios:
                - **Standard Operation:** During standard operation, sensor data from all available sources will be sent to the central server for predictions.
                - **Full Network Outage:** Treats all external sensor information as inactive and uses only the local measurements for the forecasts.

                *Standard Operation* is a Cloud solution, while during *Full Network Outage* the edge solution with lightweight models will be used.
                """
            )
        # Model scope selector
        st.radio(
            "Model scope",
            options=["Standard operation", "Full network outage"],
            index=0,
            horizontal=True,
            key="model_scope_selector",
            on_change=_on_scope_change,
            help=(
                "Standard operation: Uses all available sensor information within the network. "
                "Full network outage: Treats all external sensor information as inactive and use only local measurements for the forecasts."
            ),
        )
        # Deferred rerun after scope change (avoids warning about replacing element during event handling)
        if st.session_state.get("_scope_changed", False):
            st.session_state._scope_changed = False
            st.rerun(scope="app")
        
        is_local = st.session_state.get("model_scope_selector", "Standard operation") == "Full network outage"
        is_global = not is_local
        # Local model selector (only visible in Local mode)
        if is_local:
            st.radio(
                "Local model",
                options=["LSTM", "Transformer"],
                index=0,
                horizontal=True,
                key="local_model_selector",
                help="Choose which local model's predictions to visualize.",
            )
        elif is_global:
            st.radio(
                "Global model",
                options=["TFT", "LSTM"],
                index=0,
                horizontal=True,
                key="global_model_selector",
                help="Choose which global model's predictions to visualize (12-step ahead).",
            )

# Initialize chart components
simulation_chart = SimulationChart(
    key="pydeck_simulation_chart",
    interactive=True
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

    # Read configuration from session_state
    model_scope = st.session_state.get("model_scope_selector", "Standard operation")
    is_local_mode = (model_scope == "Full network outage")
    is_global_mode = not is_local_mode
    local_model_choice = st.session_state.get("local_model_selector", "LSTM")
    global_model_choice = st.session_state.get("global_model_selector", "TFT")

    # Current data display at the top
    with st.container(border=True):
        st.markdown("<div class='panel-header'>📊 Current Data Point</div>", unsafe_allow_html=True)
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

            # Use a stable key to avoid Streamlit recreating the element every render (reduces layout jumps)
            st.pydeck_chart(
                deck,
                use_container_width=True,
                key="enhanced_pydeck_map",
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
                iteration=None,
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
                is_playing=st.session_state.get("is_playing", False),
            )
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Render static header (runs once at startup)
render_static_header()

# Initialize session state
init_session_state()

# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]
map_data = load_map_data()

# Calculate average hours per step for speed calculation
total_data_duration = (vierlinden_data.index[-1] - vierlinden_data.index[0]).total_seconds() / 3600
avg_hours_per_step = total_data_duration / max(1, len(vierlinden_data) - 1)
max_index = len(vierlinden_data) - 1

# One-time app initialization to ensure UI starts at first time step (index 0)
# Some fragment-based reruns can cause the slider to appear at position 2 (index 1) on first load;
# enforce a single-run correction so users see the first timestep initially.
if not st.session_state.get("app_init_done", False):
    try:
        # Force first index
        st.session_state.current_idx = 0
        # last_idx is 1-based slider mirror
        st.session_state.last_idx = 1
        # Ensure slider versions are initialized
        st.session_state.time_slider_ver = st.session_state.get("time_slider_ver", 0)
        st.session_state.rain_slider_ver = st.session_state.get("rain_slider_ver", 0)
    except Exception:
        pass
    st.session_state.app_init_done = True

# =============================================================================
# TIME NAVIGATION FRAGMENT (runs on user interaction)
# =============================================================================

@st.fragment(run_every=st.session_state.get("current_update_time_interval", None))  # Auto-update at specified interval for progress bar
def time_navigation_fragment():
    """Fragment for time navigation controls - reruns on user interaction and at specified interval for progress."""
    
    # Workaround: after the first fragment reload, force a single "Back"-style step.
    # This addresses an initialization race where the UI can start on step 2 (index 1).
    if not st.session_state.get("first_step_fix_applied", False):
        st.session_state.first_step_fix_applied = True
        _step_backward()
        st.session_state.last_idx = st.session_state.current_idx + 1  # slider is 1-based
        st.session_state.time_slider_ver = st.session_state.get("time_slider_ver", 0) + 1
        st.session_state.rain_slider_ver = st.session_state.get("rain_slider_ver", 0) + 1
        st.session_state.rain_suppress = True
        st.rerun(scope="app")
        return
    
    left_col, right_col = st.columns([1.5, 6], gap="small")
    
    # Left: Configuration panel
    with left_col:
        render_dashboard_config_panel()
    
    # Right: Time navigation controls
    with right_col:
        with st.container(border=True):
            st.markdown("<div class='panel-header'>⏱️ Time Navigation</div>", unsafe_allow_html=True)
            
            # Control buttons (matching original TimeSliderLive pattern)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write("")
                # Disable Back button when at the beginning or playing (to prevent double-step issues)
                back_disabled = st.session_state.is_playing or st.session_state.current_idx <= 0
                if st.button("Back", key="nav_back", icon="⏪", use_container_width=True, disabled=back_disabled):
                    _step_backward()
            
            with col2:
                st.write("")
                # Separate Play/Pause buttons (like original)
                if st.session_state.is_playing:
                    if st.button("**Pause**", key="nav_pause_btn", type="primary", icon="⏸️", use_container_width=True):
                        st.session_state.is_playing = False
                        st.session_state.current_update_time_interval = None
                        st.rerun(scope="app")
                else:
                    if st.button("**Play**", key="nav_play_btn", type="primary", icon="▶️", use_container_width=True):
                        st.session_state.is_playing = True
                        st.session_state.current_update_time_interval = st.session_state.get("next_update_time_interval", 1.0)  # Restore last interval or default to 1s
                        st.session_state.last_update_time = time.time()
                        st.session_state.accumulated_time = 0.0
                        st.session_state.skip_autoadvance = True
                        st.rerun(scope="app")
            
            with col3:
                st.write("")
                # Disable Forward button when at the end or playing (to prevent double-step issues)
                forward_disabled = st.session_state.is_playing or st.session_state.current_idx >= max_index
                if st.button("Forward", key="nav_forward", icon="⏩", use_container_width=True, disabled=forward_disabled):
                    # Like original _step_forward() - reset timing state
                    new_idx = min(st.session_state.current_idx + 1, max_index)
                    st.session_state.current_idx = new_idx
                    st.session_state.last_update_time = time.time()
                    st.session_state.accumulated_time = 0.0
                    st.session_state.skip_autoadvance = True
                    # If at end, stop playing
                    if new_idx >= max_index:
                        st.session_state.is_playing = False
            
            with col4:
                st.write("")
                st.write("")
                if st.session_state.is_playing:
                    st.write("**Status:** ▶️ Playing")
                else:
                    st.write("**Status:** ⏸️ Paused")
            
            with col5:
                # Speed selector - use fixed default, key handles persistence
                speed_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
                
                multiplier = st.selectbox(
                    "Speed",
                    speed_options,
                    index=5,  # Default to 10x
                    format_func=lambda x: f"{x}x",
                    key="speed_multiplier",
                )
                
                st.session_state.next_update_time_interval = min(MAX_UPDATE_INTERVAL, max(1 / multiplier, MIN_UPDATE_INTERVAL))  # Limit to min/max for stability
                # Update current interval if changed while playing
                if st.session_state.get("current_update_time_interval", None) != st.session_state.next_update_time_interval and st.session_state.is_playing:
                    st.session_state.current_update_time_interval = st.session_state.next_update_time_interval
                    st.rerun(scope="app")
            
            # Time slider (when paused) or progress bar (when playing)
            if not st.session_state.is_playing:
                # Columns for sliders (right) and metrics (left) - like original
                metric_col, slider_col = st.columns([1, 10])
                
                # Initialize time slider version if needed
                if "time_slider_ver" not in st.session_state:
                    st.session_state.time_slider_ver = 0
                
                # Initialize last_idx for change detection (1-based for slider)
                if "last_idx" not in st.session_state:
                    st.session_state.last_idx = st.session_state.current_idx + 1
                
                with slider_col:
                    # Use versioned key like original TimeSliderLive
                    # 1-based for UX, internally 0-based
                    slider_val = st.slider(
                        "Select time step",
                        min_value=1,
                        max_value=max_index + 1,
                        value=st.session_state.current_idx + 1,
                        step=1,
                        key=f"time_slider_{st.session_state.time_slider_ver}",
                    )
                
                # Detect slider change (like original) - convert 1-based to 0-based
                if slider_val != st.session_state.last_idx:
                    st.session_state.current_idx = slider_val - 1
                    st.session_state.last_idx = slider_val
                    st.session_state.rain_slider_ver = st.session_state.rain_slider_ver + 1
                    st.session_state.rain_suppress = True
                    st.session_state.skip_autoadvance = True
                    st.rerun(scope="app")
                
                # If current_idx changed externally (e.g., via Back/Forward or rainfall auto-jump),
                # bump both time_slider_ver AND rain_slider_ver to force both sliders to reinitialize
                # Detect by: slider value (1-based) doesn't match last_idx but equals current_idx + 1
                if slider_val == st.session_state.last_idx and slider_val != st.session_state.current_idx + 1:
                    st.session_state.time_slider_ver = st.session_state.time_slider_ver + 1
                    st.session_state.rain_slider_ver = st.session_state.rain_slider_ver + 1
                
                # Progress metric in left column
                with metric_col:
                    progress_pct = (st.session_state.current_idx / max(1, max_index)) * 100
                    st.metric("Progress", f"{progress_pct:.1f}%", width="stretch")
                
                # Rainfall slider (when paused)
                rain_series = pd.to_numeric(vierlinden_data[RAINFALL_COLUMN], errors="coerce") if RAINFALL_COLUMN in vierlinden_data.columns else None
                if rain_series is not None and not rain_series.dropna().empty:
                    rmin = float(rain_series.min(skipna=True))
                    rmax = float(rain_series.max(skipna=True))
                    current_rain = rain_series.iloc[st.session_state.current_idx]
                    default_r = float(current_rain) if pd.notna(current_rain) else float(max(rmin, 0.0))
                    
                    rain_metric_col, rain_slider_col, rain_tol_col = st.columns([1, 9, 1])
                    
                    with rain_slider_col:
                        rain_sel = st.slider(
                            "Select rainfall",
                            min_value=rmin,
                            max_value=rmax,
                            value=min(max(default_r, rmin), rmax),
                            step=max((rmax - rmin) / 100.0, 0.001),
                            key=f"rain_slider_{st.session_state.rain_slider_ver}",
                        )
                    
                    with rain_tol_col:
                        tol = st.number_input(
                            "Tolerance (±)",
                            min_value=0.0,
                            value=0.1,
                            step=0.05,
                            key="rain_tolerance",
                        )
                    
                    # Rainfall metric in left column
                    with rain_metric_col:
                        st.metric("Rainfall (mm)", f"{rain_sel:.2f}", width="content")
                    
                    # Auto-jump logic
                    anchor_sig = f"{rain_sel:.6f}|{tol:.6f}|{st.session_state.current_idx}"
                    
                    if st.session_state.rain_suppress:
                        st.session_state.rain_suppress = False
                        st.session_state.rain_last_sig = anchor_sig
                    elif st.session_state.rain_last_sig != anchor_sig:
                        found_idx = None
                        for pos in range(st.session_state.current_idx + 1, len(vierlinden_data)):
                            v = rain_series.iloc[pos]
                            if pd.notna(v) and abs(float(v) - float(rain_sel)) <= float(tol):
                                found_idx = pos
                                break
                        
                        wrapped = False
                        if found_idx is None:
                            for pos in range(0, st.session_state.current_idx + 1):
                                v = rain_series.iloc[pos]
                                if pd.notna(v) and abs(float(v) - float(rain_sel)) <= float(tol):
                                    found_idx = pos
                                    wrapped = True
                                    break
                        
                        if found_idx is not None:
                            st.session_state.current_idx = found_idx
                            st.session_state.rain_slider_ver = st.session_state.rain_slider_ver + 1
                            st.session_state.time_slider_ver = st.session_state.time_slider_ver + 1  # Also bump time slider
                            st.session_state.rain_suppress = True
                            st.session_state.rain_last_sig = anchor_sig
                            st.rerun(scope="fragment")  # Trigger re-render to jump to new location
                            if wrapped:
                                st.warning("Wrapped to start of dataset.")
                        else:
                            st.warning("No matching rainfall found. Increase tolerance.")
                        
                        st.session_state.rain_last_sig = anchor_sig
            else:
                # Progress bar when playing (inside time nav container, updates with fragment at 2Hz)
                prog_col1, prog_col2 = st.columns([1, 10])
                with prog_col1:
                    progress_pct = (st.session_state.current_idx / max(1, max_index)) * 100
                    st.metric("Progress", f"{progress_pct:.1f}%")
                with prog_col2:
                    st.write("")
                    st.write("")
                    st.progress(st.session_state.current_idx / max(1, max_index))
            
    # Render content when paused (inside fragment so it updates on slider change)
    if not st.session_state.is_playing:
        idx = st.session_state.current_idx
        timestamp = vierlinden_data.index[idx]
        data_row = vierlinden_data.iloc[idx]
        smooth_content_renderer(idx, timestamp, data_row, idx)


# =============================================================================
# DYNAMIC CONTENT FRAGMENT (auto-updates via run_every)
# =============================================================================

@st.fragment(run_every=st.session_state.get("current_update_time_interval", None))  # Auto-update at specified interval
def dynamic_content_fragment():
    """Fragment for dynamic content - auto-updates every UPDATE_TIME_INTERVAL seconds when playing."""
    
    # Auto-advance when playing
    if st.session_state.is_playing:
        if st.session_state.skip_autoadvance:
            st.session_state.skip_autoadvance = False
        else:
            # Time-based speed calculation
            current_time = time.time()
            elapsed_real_time = current_time - st.session_state.last_update_time
            st.session_state.last_update_time = current_time
            
            # Get speed (hours per second) - use .get() with default since not pre-initialized
            hours_per_second = st.session_state.get("speed_multiplier", 10.0)
            
            # Calculate how much data time should have passed
            data_hours_elapsed = elapsed_real_time * hours_per_second
            
            # Accumulate fractional time
            accumulated = st.session_state.accumulated_time + data_hours_elapsed
            st.session_state.accumulated_time = accumulated
            
            # Convert accumulated time to steps (allow fractional accumulation)
            steps_to_advance = int(accumulated / avg_hours_per_step)
            
            # Keep remainder for next iteration
            if steps_to_advance > 0:
                remainder = accumulated - (steps_to_advance * avg_hours_per_step)
                st.session_state.accumulated_time = remainder
                
                # Advance the index
                new_idx = st.session_state.current_idx + steps_to_advance
                if new_idx >= max_index:
                    st.session_state.current_idx = 0
                    st.session_state.last_update_time = time.time()
                    st.session_state.accumulated_time = 0.0
                    st.toast("Reached end of data. Looping back to the beginning!", icon="🔄")
                else:
                    st.session_state.current_idx = new_idx
        
        # Render content when playing (inside fragment so it updates at 2Hz)
        idx = st.session_state.current_idx
        timestamp = vierlinden_data.index[idx]
        data_row = vierlinden_data.iloc[idx]
        smooth_content_renderer(idx, timestamp, data_row, idx)
    # When paused: content is rendered in time_navigation_fragment


# =============================================================================
# RUN FRAGMENTS (controls + content rendering)
# =============================================================================

# Run time navigation fragment (handles user input + renders content when paused)
time_navigation_fragment()

# When playing, content is rendered by dynamic_content_fragment
# When paused, content is rendered inside time_navigation_fragment

# Run dynamic content fragment only when playing (auto-updates at 2Hz)
if st.session_state.is_playing:
    dynamic_content_fragment()

st.markdown("Thank you for trying this demo! For more information, questions, or suggestions, contact us at: Vipin.Singh@bht-berlin.de<br> Learn more about our research at: https://calgo-lab.de", unsafe_allow_html=True)
# Spacer preserved for page length
st.markdown("<div style='height:600px;'></div>", unsafe_allow_html=True)
