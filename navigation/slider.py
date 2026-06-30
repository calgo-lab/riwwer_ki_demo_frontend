"""Time and rainfall sliders for RIWWER ML Demo."""
import pandas as pd
import streamlit as st


def render_time_slider(max_index: int, rain_series: pd.Series | None = None) -> None:
    """Render the time slider with progress metric.
    
    Args:
        max_index: Maximum index value for the data.
        rain_series: Optional rainfall series for triggering rain slider version bump.
    """
    metric_col, slider_col = st.columns([1, 10])
    
    # Initialize time slider version if needed
    if "time_slider_ver" not in st.session_state:
        st.session_state.time_slider_ver = 0
    
    # Initialize last_idx for change detection (1-based for slider)
    if "last_idx" not in st.session_state:
        st.session_state.last_idx = st.session_state.current_idx + 1
    
    with slider_col:
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


def render_rainfall_slider(vierlinden_data: pd.DataFrame, rainfall_column: str) -> None:
    """Render the rainfall slider with auto-jump logic.
    
    Args:
        vierlinden_data: The main data DataFrame.
        rainfall_column: Name of the rainfall column.
    """
    rain_series = pd.to_numeric(vierlinden_data[rainfall_column], errors="coerce") if rainfall_column in vierlinden_data.columns else None
    
    if rain_series is None or rain_series.dropna().empty:
        return
    
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