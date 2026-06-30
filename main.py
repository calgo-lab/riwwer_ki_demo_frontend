import streamlit as st
import pandas as pd
import time
from utils.config import (
    read_data,
    TARGET_COLUMN,
    MAP_DATA,
    COORDINATES_DICT,
    CACHE_TIME,
    calculate_target_column_bounds,
    RAINFALL_COLUMN,
    LOCAL_PREDICTIONS_PATH,
    PREDICTIONS_TIME_COLUMN,
    LOCAL_LSTM_PRED_COLUMN,
    LOCAL_TRANSFORMER_PRED_COLUMN,
    GLOBAL_PREDICTIONS_PATH,
    GLOBAL_TFT_PRED_COLUMN,
    GLOBAL_LSTM_PRED_COLUMN,
    OVERFLOW_CLS_PRED_FILE_PATH
)
from utils.dynamic_map_data import load_map_data
from state import StateManager
from rendering import (
    render_static_header,
    render_dashboard_config_panel,
    smooth_content_renderer,
)
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
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables for time navigation."""
    StateManager.init()


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

# Pre-calculate map center based on bounding box (not just coordinate average)
lats = [coord[1] for coord in COORDINATES_DICT.values()]
lons = [coord[0] for coord in COORDINATES_DICT.values()]
min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2


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
        StateManager.step_backward()
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
                    StateManager.step_backward()
            
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
                        StateManager.reset_timing()
                        st.rerun(scope="app")
            
            with col3:
                st.write("")
                # Disable Forward button when at the end or playing (to prevent double-step issues)
                forward_disabled = st.session_state.is_playing or st.session_state.current_idx >= max_index
                if st.button("Forward", key="nav_forward", icon="⏩", use_container_width=True, disabled=forward_disabled):
                    # Like original _step_forward() - reset timing state
                    new_idx = StateManager.step_forward(max_index)
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
        smooth_content_renderer(
            idx, timestamp, data_row, idx,
            vierlinden_data, map_data, local_preds, global_preds,
            overflow_cls_predictions, y_axis_bounds, center_lat, center_lon
        )


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
                    StateManager.reset_timing()
                    st.toast("Reached end of data. Looping back to the beginning!", icon="🔄")
                else:
                    st.session_state.current_idx = new_idx
        
        # Render content when playing (inside fragment so it updates at 2Hz)
        idx = st.session_state.current_idx
        timestamp = vierlinden_data.index[idx]
        data_row = vierlinden_data.iloc[idx]
        smooth_content_renderer(
            idx, timestamp, data_row, idx,
            vierlinden_data, map_data, local_preds, global_preds,
            overflow_cls_predictions, y_axis_bounds, center_lat, center_lon
        )
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
