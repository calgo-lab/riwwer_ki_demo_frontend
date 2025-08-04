import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Callable, Any
import plotly.graph_objects as go


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
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button("⏮️ Start", key=f"{self.session_key}_start"):
                self._reset_to_start()

        with col2:
            if st.button("⏪ Back", key=f"{self.session_key}_back"):
                self._step_backward()

        with col3:
            # Get current state and show appropriate button
            current_playing_state = st.session_state.get(
                f"{self.session_key}_is_playing", False
            )

            if current_playing_state:
                # Currently playing - show pause button
                if st.button("⏸️ Pause", key=f"{self.session_key}_pause_btn"):
                    st.session_state[f"{self.session_key}_is_playing"] = False
                    # Force immediate refresh to show the new button state
                    st.rerun()
            else:
                # Currently paused - show play button
                if st.button("▶️ Play", key=f"{self.session_key}_play_btn"):
                    st.session_state[f"{self.session_key}_is_playing"] = True
                    # Reset timing when starting to play
                    st.session_state[f"{self.session_key}_last_update_time"] = (
                        time.time()
                    )
                    st.session_state[f"{self.session_key}_accumulated_time"] = 0.0
                    # Force immediate refresh to show the new button state
                    st.rerun()

        with col4:
            if st.button("⏩ Forward", key=f"{self.session_key}_forward"):
                self._step_forward()

        with col5:
            if st.button("⏭️ End", key=f"{self.session_key}_end"):
                self._reset_to_end()

        with col6:
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
        hours_per_second: float = 24.0,
        updates_per_second: float = 2.0,
        show_controls: bool = True,
        show_progress: bool = True,
        max_iterations: int = 1000,
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
        # Show controls at the top
        if show_controls:
            st.subheader("Time Navigation Controls")
            self.render_controls()

        # Create placeholder for the main content
        placeholder = st.empty()

        # Calculate sleep time between updates
        sleep_time = 1.0 / updates_per_second

        # Main update loop
        for iteration in range(max_iterations):
            # Get current speed from controls (updated each iteration to reflect button clicks)
            effective_speed = st.session_state[f"{self.session_key}_speed_multiplier"]

            with placeholder.container():
                # Get current data
                current_idx, current_timestamp, current_data = self.get_current_data()

                # Show progress information
                if show_progress:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Index:** {current_idx + 1} / {len(self.data)}")
                    with col2:
                        st.write(
                            f"**Time:** {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    with col3:
                        status = "▶️ Playing" if self._is_playing() else "⏸️ Paused"
                        st.write(f"**Status:** {status} ({effective_speed}x)")

                # Progress bar
                if show_progress:
                    progress = current_idx / max(1, self.max_index)
                    st.progress(progress)

                # Render the main content using the provided renderer
                try:
                    content_renderer(
                        current_idx, current_timestamp, current_data, iteration
                    )
                except Exception as e:
                    st.error(f"Error in content renderer: {e}")

                # Handle autoplay advancement
                if self._is_playing():
                    steps_to_advance = self._calculate_time_step(
                        effective_speed,  # Use speed from controls directly
                        updates_per_second,
                    )
                    if steps_to_advance > 0:
                        self._step_forward(steps_to_advance)

                # Sleep to control update rate
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
