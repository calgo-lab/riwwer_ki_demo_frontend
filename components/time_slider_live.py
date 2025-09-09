import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Callable, Any
import plotly.graph_objects as go
try:
    # Optional: used for rainfall event selector
    from utils.config import RAINFALL_COLUMN as _RAIN_COL
except Exception:
    _RAIN_COL = None


class TimeSliderLive:
    """
    A high-performance time slider using st.empty() placeholder pattern.

    This approach is much more efficient than st.rerun() as it only updates
    the content within the placeholder container, not the entire page.

    Features:
    - Continuous autoplay without page refreshes
    - Interactive controls (play/pause, navigation)
    - Real-time data display
    - Configurable speed and update rate
    - Custom content renderer support

    Example:
        ```python
        from components.time_slider_live import TimeSliderLive

        def render_content(idx, timestamp, data_row):
            st.write(f"Current: {timestamp}")
            st.write(f"Value: {data_row['target_column']}")

        slider = TimeSliderLive(data)
        slider.run_live_dashboard(
            content_renderer=render_content,
            hours_per_second=24.0,
            updates_per_second=2.0
        )
        ```
    """

    def __init__(self, data: pd.DataFrame, session_key: str = "time_slider_live"):
        """
        Initialize the TimeSliderLive component.

        Args:
            data: DataFrame with DatetimeIndex
            session_key: Unique key for session state variables
        """
        self.data = data
        self.session_key = session_key
        self.max_index = len(data) - 1
        self.min_index = 0

        # Validate input data
        if data.empty:
            raise ValueError("Data cannot be empty")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Store timestamps for efficient access
        self.timestamps = data.index.tolist()

        # Initialize session state for controls
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables."""
        keys = {
            f"{self.session_key}_current_idx": 0,
            f"{self.session_key}_is_playing": False,
            f"{self.session_key}_speed_multiplier": 10.0,  # Default to 10x speed
            f"{self.session_key}_last_update_time": time.time(),
            f"{self.session_key}_accumulated_time": 0.0,
        }

        for key, default_value in keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _get_current_index(self) -> int:
        """Get the current index."""
        return st.session_state[f"{self.session_key}_current_idx"]

    def _set_current_index(self, idx: int) -> None:
        """Set the current index with bounds checking."""
        idx = max(self.min_index, min(idx, self.max_index))
        st.session_state[f"{self.session_key}_current_idx"] = idx
        # Reset timing accumulators when index is set programmatically to prevent double step
        st.session_state[f"{self.session_key}_last_update_time"] = time.time()
        st.session_state[f"{self.session_key}_accumulated_time"] = 0.0
        st.session_state[f"{self.session_key}_skip_autoadvance"] = True

    def _is_playing(self) -> bool:
        """Check if autoplay is active."""
        return st.session_state.get(f"{self.session_key}_is_playing", False)

    def _toggle_play_pause(self) -> None:
        """Toggle play/pause state."""
        current_state = st.session_state[f"{self.session_key}_is_playing"]
        st.session_state[f"{self.session_key}_is_playing"] = not current_state

        # Reset timing when starting
        if not current_state:
            st.session_state[f"{self.session_key}_last_update_time"] = time.time()
            st.session_state[f"{self.session_key}_accumulated_time"] = 0.0

    def _step_forward(self, steps: int = 1) -> None:
        """Step forward by specified number of steps."""
        current_idx = self._get_current_index()
        new_idx = min(current_idx + steps, self.max_index)
        self._set_current_index(new_idx)
        # Pause if we hit the end
        if new_idx >= self.max_index:
            st.session_state[f"{self.session_key}_is_playing"] = False

    def _step_backward(self, steps: int = 1) -> None:
        """Step backward by specified number of steps."""
        current_idx = self._get_current_index()
        new_idx = max(current_idx - steps, self.min_index)
        self._set_current_index(new_idx)

    def _reset_to_start(self) -> None:
        """Reset to the beginning."""
        self._set_current_index(self.min_index)
        st.session_state[f"{self.session_key}_is_playing"] = False

    def _reset_to_end(self) -> None:
        """Reset to the end."""
        self._set_current_index(self.max_index)
        st.session_state[f"{self.session_key}_is_playing"] = False

    def _calculate_time_step(
        self, hours_per_second: float, updates_per_second: float
    ) -> int:
        """
        Calculate how many data points to advance based on time progression.

        Args:
            hours_per_second: How many data hours should pass per real second
            updates_per_second: How many times per second to update

        Returns:
            Number of indices to advance
        """
        current_time = time.time()
        last_update_time = st.session_state[f"{self.session_key}_last_update_time"]

        # Calculate elapsed real time
        elapsed_real_time = current_time - last_update_time
        st.session_state[f"{self.session_key}_last_update_time"] = current_time

        # Calculate how much data time should have passed
        data_hours_elapsed = elapsed_real_time * hours_per_second

        # Accumulate fractional time
        accumulated = st.session_state[f"{self.session_key}_accumulated_time"]
        accumulated += data_hours_elapsed
        st.session_state[f"{self.session_key}_accumulated_time"] = accumulated

        # Calculate average time between data points (in hours)
        if len(self.data) > 1:
            total_data_duration = (
                self.timestamps[-1] - self.timestamps[0]
            ).total_seconds() / 3600
            avg_hours_per_step = total_data_duration / (len(self.data) - 1)
        else:
            avg_hours_per_step = 1.0

        # Convert accumulated time to steps
        steps_to_advance = int(accumulated / avg_hours_per_step)

        # Keep remainder for next iteration
        if steps_to_advance > 0:
            st.session_state[f"{self.session_key}_accumulated_time"] = accumulated - (
                steps_to_advance * avg_hours_per_step
            )

        return steps_to_advance

    def render_controls(self) -> None:
        """Render the control buttons."""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # Pad to center it
            st.write("")
            if st.button("Back", key=f"{self.session_key}_back", icon="⏪"):
                self._step_backward()
                st.session_state[f"{self.session_key}_skip_autoadvance"] = True

        with col2:
            # Pad to center it
            st.write("")
            # Get current state and show appropriate button
            current_playing_state = st.session_state.get(
                f"{self.session_key}_is_playing", False
            )

            if current_playing_state:
                # Currently playing - show pause button
                if st.button("**Pause**", key=f"{self.session_key}_pause_btn", type="primary", icon="⏸️"):
                    st.session_state[f"{self.session_key}_is_playing"] = False
                    # Force immediate refresh to show the new button state
                    st.rerun()
            else:
                # Currently paused - show play button
                if st.button("**Play**", key=f"{self.session_key}_play_btn", type="primary", icon="▶️"):
                    st.session_state[f"{self.session_key}_is_playing"] = True
                    # Reset timing when starting to play
                    st.session_state[f"{self.session_key}_last_update_time"] = (
                        time.time()
                    )
                    st.session_state[f"{self.session_key}_accumulated_time"] = 0.0
                    # Ensure first playing iteration doesn't auto-advance immediately
                    st.session_state[f"{self.session_key}_skip_autoadvance"] = True
                    # Force immediate refresh to show the new button state
                    st.rerun()

        with col3:
            # Pad to center it
            st.write("")
            if st.button("Forward", key=f"{self.session_key}_forward", icon="⏩"):
                self._step_forward()
                st.session_state[f"{self.session_key}_skip_autoadvance"] = True

        with col4:
            # Pad to center it
            st.write("")
            st.write("")
            if self._is_playing():
                st.write(f"**Status:** ▶️ Playing")
            else:
                st.write(f"**Status:** ⏸️ Paused")

        with col5:
            # Speed control with better options
            speed_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

            # Get the current speed from session state to maintain selection across reruns
            current_speed = st.session_state.get(
                f"{self.session_key}_speed_multiplier", 10.0
            )

            # Find the index of the current speed, default to 10x if not found
            try:
                current_index = speed_options.index(current_speed)
            except ValueError:
                current_index = 5  # Default to 10x

            speed = st.selectbox(
                "Speed",
                speed_options,
                index=current_index,  # Use current speed index instead of fixed default
                format_func=lambda x: f"{x}x",
                key=f"{self.session_key}_speed",
            )

            # Update session state when speed changes
            st.session_state[f"{self.session_key}_speed_multiplier"] = speed

    def get_current_data(self) -> Tuple[int, pd.Timestamp, pd.Series]:
        """
        Get current data without rendering.

        Returns:
            Tuple of (current_index, current_timestamp, current_data_row)
        """
        current_idx = self._get_current_index()
        current_timestamp = self.timestamps[current_idx]
        current_data = self.data.iloc[current_idx]
        return current_idx, current_timestamp, current_data

    def run_live_dashboard(
        self,
        content_renderer: Callable[[int, pd.Timestamp, pd.Series, int], None],
        updates_per_second: float = 2.0,
        show_controls: bool = True,
        max_iterations: int = 1000,
        left_of_nav_renderer: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Run the live dashboard with continuous updates.

        Args:
            content_renderer: Function to render the main content (idx, timestamp, data_row, iteration)
            hours_per_second: How many data hours should pass per real second
            updates_per_second: How many times per second to update
            show_controls: Whether to show control buttons
            show_progress: Whether to show progress information
            max_iterations: Maximum number of iterations (safety limit)
        """
        # Create a single panel that contains controls and the dynamic timeline UI
        # Create two sibling panels side-by-side: left (config) and right (time navigation)
        left_col, right_col = st.columns([1.5, 6], gap="small")

        # Left: external renderer builds its own bordered panel
        with left_col:
            if callable(left_of_nav_renderer):
                try:
                    left_of_nav_renderer()
                except Exception:
                    pass

        # Right: our own bordered Time Navigation panel
        with right_col:
            nav_panel = st.container(border=True)
            with nav_panel:
                st.markdown("<div class='panel-header'>⏱️ Time Navigation</div>", unsafe_allow_html=True)
                # Static controls (render once per run to avoid duplicate widget keys)
                if show_controls:
                    self.render_controls()
                # Placeholder for dynamic nav UI (slider when paused, progress when playing)
                nav_placeholder = st.empty()

        # Separate placeholder for the main content area (outside the nav panel)
        content_placeholder = st.empty()

        # Hard wipe when switching modes (paused <-> playing) to avoid stacked/dimmed components
        mode_key = f"{self.session_key}_last_mode"
        current_mode = "playing" if self._is_playing() else "paused"
        if st.session_state.get(mode_key) != current_mode:
            try:
                nav_placeholder.empty()
            except Exception:
                pass
            try:
                content_placeholder.empty()
            except Exception:
                pass
            st.session_state[mode_key] = current_mode
            st.rerun()
            return

        # Calculate sleep time between updates
        sleep_time = 1.0 / updates_per_second

        # If paused, render a single snapshot with an interactive slider and return
        if not self._is_playing():
            effective_speed = st.session_state[f"{self.session_key}_speed_multiplier"]
            # 1) Navigation panel content (controls + sliders)
            with nav_placeholder.container():
                # Interactive slider while paused (render once per run)
                baseline_idx = self._get_current_index()  # 0-based internal index
                slider_key = f"{self.session_key}_idx_slider"
                time_slider_ver_key = f"{self.session_key}_idx_slider_ver"
                last_idx_key = f"{self.session_key}_last_idx"

                # Prepare rainfall slider shared keys (used for sync and versioning)
                rain_slider_base_key = f"{self.session_key}_rain_slider"
                rain_slider_ver_key = f"{self.session_key}_rain_slider_ver"
                rain_tol_key = f"{self.session_key}_rain_tol"
                rain_last_key = f"{self.session_key}_rain_last"
                rain_suppress_key = f"{self.session_key}_rain_suppress"
                if time_slider_ver_key not in st.session_state:
                    st.session_state[time_slider_ver_key] = 0
                if rain_slider_ver_key not in st.session_state:
                    st.session_state[rain_slider_ver_key] = 0
                if rain_suppress_key not in st.session_state:
                    st.session_state[rain_suppress_key] = False

                # Initialize last_idx the first time
                if last_idx_key not in st.session_state:
                    st.session_state[last_idx_key] = baseline_idx
                    # On very first render, don't auto-search rainfall; we'll set anchor and wait for user action
                    st.session_state[rain_suppress_key] = True

                # If index changed via buttons or other controls, sync the slider to baseline
                if st.session_state[last_idx_key] != baseline_idx:
                    # Bump time slider version so it re-instantiates with new default
                    st.session_state[time_slider_ver_key] = st.session_state.get(time_slider_ver_key, 0) + 1
                    # Prevent rainfall auto-search on this rerun and refresh rainfall slider default
                    st.session_state[rain_suppress_key] = True
                    st.session_state[rain_slider_ver_key] = st.session_state.get(rain_slider_ver_key, 0) + 1

                # Columns for sliders (right) and metrics (left)
                col1, col2 = st.columns([1, 10])

                # Render the time step slider first so we can reflect its value in metrics
                with col2:
                    # Slider is 1-based for UX, mapped to 0-based internally
                    slider_val = st.slider(
                        "Select time step",
                        min_value=1,
                        max_value=len(self.data),
                        value=int(baseline_idx + 1),
                        step=1,
                        key=f"{slider_key}_{st.session_state[time_slider_ver_key]}",
                    )
                    
                    sel_idx0 = max(self.min_index, min(self.max_index, int(slider_val) - 1))
                    
                    # Immediately apply slider selection to current index so render reflects it now
                    if sel_idx0 != baseline_idx:
                        # Bump rainfall slider version so it will reset to new rainfall on rerun
                        st.session_state[rain_slider_ver_key] = st.session_state.get(rain_slider_ver_key, 0) + 1
                        st.session_state[rain_suppress_key] = True
                        self._set_current_index(sel_idx0)
                        # Force a rerun to re-instantiate widgets with up-to-date defaults
                        st.rerun()

                # Metrics reflect the current (possibly updated) index
                with col1:
                    current_for_metric = self._get_current_index()
                    progress_pct = (current_for_metric / max(1, len(self.data) - 1)) * 100
                    st.metric("Progress", f"{progress_pct:.1f}%")

                # Optional rainfall event selector (only when paused)
                # Placed directly under the time step slider
                rain_col_name = None
                if _RAIN_COL is not None and isinstance(_RAIN_COL, str) and _RAIN_COL in self.data.columns:
                    rain_col_name = _RAIN_COL

                if rain_col_name is not None:
                    # Build rainfall selector first, then show its metric for perfect sync
                    with col2:
                        rain_series = pd.to_numeric(self.data[rain_col_name], errors="coerce")
                        if not rain_series.dropna().empty:
                            rmin = float(rain_series.min(skipna=True))
                            rmax = float(rain_series.max(skipna=True))
                            # Reasonable defaults
                            default_r = float(rain_series.iloc[baseline_idx]) if pd.notna(rain_series.iloc[baseline_idx]) else float(max(rmin, 0.0))
                            # Slider and tolerance controls
                            rs_col, tol_col = st.columns([4, 1])

                            with rs_col:
                                rain_sel = st.slider(
                                    "Select rainfall",
                                    min_value=rmin,
                                    max_value=rmax,
                                    value=min(max(default_r, rmin), rmax),
                                    step=max((rmax - rmin) / 100.0, 0.001),
                                    key=f"{rain_slider_base_key}_{st.session_state[rain_slider_ver_key]}",
                                )
                            with tol_col:
                                tol = st.number_input(
                                    "Tolerance",
                                    min_value=0.0,
                                    value=0.1,
                                    step=0.05,
                                    key=rain_tol_key,
                                    help="Allowed difference between rainfall and selected value",
                                )

                            # Now show the selected rainfall metric in the left column
                            with col1:
                                st.metric("Rainfall (mm)", f"{rain_sel:.2f}")

                            # Only perform a jump once per (value, tolerance, anchor_idx)
                            anchor_sig = f"{rain_sel:.6f}|{tol:.6f}|{baseline_idx}"
                            if st.session_state.get(rain_suppress_key):
                                # Skip search once when we've just synced the rainfall slider programmatically
                                st.session_state[rain_suppress_key] = False
                                st.session_state[rain_last_key] = anchor_sig
                            elif st.session_state.get(rain_last_key) != anchor_sig:
                                # Search forward first
                                found_idx = None
                                for pos in range(baseline_idx + 1, self.max_index + 1):
                                    v = rain_series.iloc[pos]
                                    if pd.notna(v) and abs(float(v) - float(rain_sel)) <= float(tol):
                                        found_idx = pos
                                        break
                                wrapped = False
                                if found_idx is None:
                                    # Wrap-around search from start to current
                                    for pos in range(self.min_index, baseline_idx + 1):
                                        v = rain_series.iloc[pos]
                                        if pd.notna(v) and abs(float(v) - float(rain_sel)) <= float(tol):
                                            found_idx = pos
                                            wrapped = True
                                            break

                                if found_idx is not None:
                                    self._set_current_index(found_idx)
                                    baseline_idx = found_idx
                                    # After jumping, bump version so the rainfall slider value resets; then rerun
                                    st.session_state[rain_slider_ver_key] = st.session_state.get(rain_slider_ver_key, 0) + 1
                                    st.session_state[rain_suppress_key] = True
                                    st.session_state[rain_last_key] = anchor_sig
                                    st.rerun()
                                    if wrapped:
                                        st.warning("Wrapped to start of dataset to find next matching rainfall.")
                                else:
                                    st.warning("No time step with rainfall within tolerance found in entire dataset. *Please increase the tolerance* (see to the right of the slider).")

                                st.session_state[rain_last_key] = anchor_sig
                        else:
                            st.info("No valid rainfall data available for event selection.")
                else:
                    # Uncomment if you'd like to inform about missing config or column
                    # st.info("Rainfall event selector unavailable: rainfall column not configured or missing.")
                    pass

                # Update last_idx to the index we are rendering now (after any jumps)
                st.session_state[last_idx_key] = baseline_idx

            # 2) Render main content area separately (outside nav panel)
            current_idx, current_timestamp, current_data = self.get_current_data()
            with content_placeholder.container():
                try:
                    content_renderer(current_idx, current_timestamp, current_data, 0)
                except Exception as e:
                    st.error(f"Error in content renderer: {e}")
            return

        # Main update loop (playing): progress bar only, no interactive widgets here
        for iteration in range(max_iterations):
            effective_speed = st.session_state[f"{self.session_key}_speed_multiplier"]

            # If playing state flipped to paused externally (e.g., hit end last iteration),
            # immediately rerun to render paused UI (with sliders) instead of progress bar.
            if not self._is_playing():
                st.rerun()
                return

            # 1) Navigation panel content while playing (controls + progress bar)
            with nav_placeholder.container():
                current_idx, current_timestamp, current_data = self.get_current_data()
                col1, col2 = st.columns([1, 10])
                progress = current_idx / max(1, self.max_index)
                with col1:
                    st.metric("Progress", f"{progress*100:.1f}%")
                with col2:
                    st.write("")
                    st.write("")
                    st.progress(progress)

            # 2) Main content area (outside nav panel)
            with content_placeholder.container():
                try:
                    content_renderer(current_idx, current_timestamp, current_data, iteration)
                except Exception as e:
                    st.error(f"Error in content renderer: {e}")

                if self._is_playing():
                    # If a manual navigation or fresh play just occurred, skip one auto-advance to avoid double step
                    if st.session_state.get(f"{self.session_key}_skip_autoadvance"):
                        st.session_state[f"{self.session_key}_skip_autoadvance"] = False
                    else:
                        steps_to_advance = self._calculate_time_step(effective_speed, updates_per_second)
                        if steps_to_advance > 0:
                            self._step_forward(steps_to_advance)
                            # If we hit the end inside _step_forward, state flips to paused.
                            # Rerun immediately so the paused UI (with sliders) appears.
                            if not self._is_playing():
                                st.rerun()
                                return
                else:
                    # If user paused during loop, rerun to render paused view immediately
                    st.rerun()
                    return

            time.sleep(sleep_time)

    def run_simple_loop(
        self,
        target_column: str,
        hours_per_second: float = 24.0,
        updates_per_second: float = 2.0,
        show_chart: bool = True,
        max_iterations: int = 1000,
    ) -> None:
        """
        Run a simple loop with default content rendering.

        Args:
            target_column: Column name to display and chart
            hours_per_second: How many data hours should pass per real second
            updates_per_second: How many times per second to update
            show_chart: Whether to show the line chart
            max_iterations: Maximum number of iterations
        """

        def default_content_renderer(
            idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int
        ):
            # Display current values
            st.subheader("Current Data Point")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Value", f"{data_row[target_column]:.2f}")
            with col2:
                st.metric("Timestamp", timestamp.strftime("%H:%M:%S"))

            # Show chart if requested
            if show_chart:
                # Create a simple line chart showing recent history
                window_size = min(100, len(self.data))
                start_idx = max(0, idx - window_size + 1)
                end_idx = idx + 1

                chart_data = self.data.iloc[start_idx:end_idx]

                if not chart_data.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data[target_column],
                            mode="lines",
                            name=target_column,
                            line=dict(color="blue", width=2),
                        )
                    )

                    # Highlight current point
                    fig.add_trace(
                        go.Scatter(
                            x=[timestamp],
                            y=[data_row[target_column]],
                            mode="markers",
                            name="Current",
                            marker=dict(color="red", size=10),
                        )
                    )

                    fig.update_layout(
                        title=f"{target_column} - Last {window_size} Points",
                        xaxis_title="Time",
                        yaxis_title=target_column,
                        height=400,
                    )

                    # Create unique key based on current timestamp to avoid conflicts
                    chart_key = f"live_chart_iter_{iteration}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)

            # Show raw data
            with st.expander("View Current Row Data"):
                st.dataframe(data_row.to_frame().T)

        self.run_live_dashboard(
            content_renderer=default_content_renderer,
            hours_per_second=hours_per_second,
            updates_per_second=updates_per_second,
        )
