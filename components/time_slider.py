import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time


class TimeSlider:
    """
    A reusable time slider component with autoplay functionality for Streamlit apps.

    Features:
    - Interactive time navigation with slider
    - Autoplay with configurable speed
    - Button controls for navigation
    - Debounced button clicks to prevent rapid firing
    - Memory-optimized for large datasets
    - Non-uniform time interval detection
    - Graceful error handling

    Example:
        ```python
        import pandas as pd
        from components.time_slider import TimeSlider

        # Create time-indexed DataFrame
        data = pd.DataFrame(
            {'value': range(100)},
            index=pd.date_range('2023-01-01', periods=100, freq='H')
        )

        # Create and render slider
        slider = TimeSlider(data, session_key="my_slider")
        idx, timestamp, row = slider.render(
            label="Select Time",
            hours_per_second=24.0,  # 1 day per second
            renders_per_second=4.0   # 4 updates per second
        )
        ```
    """

    def __init__(self, data: pd.DataFrame, session_key: str = "time_slider"):
        """
        Initialize the TimeSlider component.

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
            raise ValueError("DataFrame cannot be empty")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        # Check for non-uniform time intervals and warn if detected
        if len(data) > 2:
            deltas = data.index[1:] - data.index[:-1]
            unique_deltas = deltas.nunique()
            # Only warn if there are significant irregularities (more than DST transitions)
            if (
                unique_deltas > 3
            ):  # Allow for 1h, 2h, and 0h intervals (DST transitions)
                st.warning(
                    "⚠️ Non-uniform time intervals detected. Autoplay timing may be inconsistent."
                )
            elif unique_deltas > 1:
                # Show info about DST transitions instead of warning
                non_hourly_count = (deltas != pd.Timedelta(hours=1)).sum()
                st.info(
                    f"ℹ️ Dataset contains {non_hourly_count} daylight saving time transitions."
                )

        # Store only timestamps for memory optimization with large datasets
        self.timestamps = data.index.tolist()
        self.formatted_timestamps = [
            self._format_timestamp(ts) for ts in self.timestamps
        ]

        # Initialize session state
        slider_key = f"{session_key}_slider"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0
        # store slider key for internal methods
        self.slider_key = slider_key
        if f"{session_key}_autoplay" not in st.session_state:
            st.session_state[f"{session_key}_autoplay"] = False
        # accumulator for fractional steps during autoplay
        if f"{session_key}_accum" not in st.session_state:
            st.session_state[f"{session_key}_accum"] = 0.0
        # debouncing for button clicks
        if f"{session_key}_last_action_time" not in st.session_state:
            st.session_state[f"{session_key}_last_action_time"] = 0.0
        # button action flags
        if f"{session_key}_button_action" not in st.session_state:
            st.session_state[f"{session_key}_button_action"] = None

    def _get_current_index(self) -> int:
        """Get the current slider index."""
        return st.session_state[self.slider_key]

    def _set_current_index(self, value: int) -> None:
        """Set the current slider index."""
        st.session_state[self.slider_key] = value

    def _is_autoplay_active(self) -> bool:
        """Check if autoplay is active."""
        return st.session_state[f"{self.session_key}_autoplay"]

    def _toggle_autoplay_direct(self) -> None:
        """Directly toggle autoplay state - used as button callback."""
        current_state = st.session_state[f"{self.session_key}_autoplay"]
        st.session_state[f"{self.session_key}_autoplay"] = not current_state

        if not current_state:  # Starting autoplay
            # Reset accumulator when starting autoplay
            st.session_state[f"{self.session_key}_accum"] = 0.0
        else:  # Stopping autoplay
            # Clear accumulator when stopping
            st.session_state[f"{self.session_key}_accum"] = 0.0

    def _stop_autoplay(self) -> None:
        """Stop autoplay when user manually changes the slider."""
        st.session_state[f"{self.session_key}_autoplay"] = False
        st.session_state[f"{self.session_key}_accum"] = 0.0

    def _button_callback(self, action: str, steps: int = 1) -> None:
        """Callback for button actions that sets a flag for processing."""
        if self._debounce_action():
            st.session_state[f"{self.session_key}_button_action"] = (action, steps)

    def _debounce_action(self, min_interval: float = 0.1) -> bool:
        """
        Simple debouncing to prevent rapid successive actions.

        Args:
            min_interval: Minimum time between actions in seconds

        Returns:
            True if action should proceed, False if it should be debounced
        """
        current_time = time.time()
        last_action_time = st.session_state[f"{self.session_key}_last_action_time"]

        if current_time - last_action_time >= min_interval:
            st.session_state[f"{self.session_key}_last_action_time"] = current_time
            return True
        return False

    def _format_timestamp(self, timestamp: pd.Timestamp) -> str:
        """Format timestamp for display."""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def _get_timestamp_options(self) -> Tuple[list, list]:
        """Get timestamp options for the slider."""
        return self.timestamps, self.formatted_timestamps

    def reset_to_start(self) -> None:
        """Reset slider to the beginning."""
        st.session_state[self.slider_key] = self.min_index
        self._stop_autoplay()

    def reset_to_end(self) -> None:
        """Reset slider to the end."""
        st.session_state[self.slider_key] = self.max_index
        self._stop_autoplay()

    def jump_to_timestamp(self, target_timestamp: pd.Timestamp) -> bool:
        """
        Jump to the closest timestamp in the data.

        Args:
            target_timestamp: The timestamp to jump to

        Returns:
            True if successful, False if timestamp not found or invalid
        """
        try:
            # Find the closest timestamp
            time_diff = abs(self.data.index - target_timestamp)
            closest_idx = time_diff.argmin()
            st.session_state[self.slider_key] = closest_idx
            self._stop_autoplay()
            return True
        except Exception:
            return False

    def get_current_data(self) -> Tuple[int, pd.Timestamp, pd.Series]:
        """
        Get current slider state without rendering.

        Returns:
            Tuple of (current_index, current_timestamp, current_data_row)
        """
        current_idx = self._get_current_index()
        current_timestamp = self.timestamps[current_idx]
        current_data = self.data.iloc[current_idx]
        return current_idx, current_timestamp, current_data

    def render(
        self,
        label: str = "Time Event",
        show_controls: bool = True,
        show_current_info: bool = True,
        hours_per_second: float = 1.0,
        renders_per_second: float = 2.0,
    ) -> Tuple[int, pd.Timestamp]:
        """
        Render the time slider component with intelligent time-based autoplay.

        Args:
            label: Label for the slider
            show_controls: Whether to show prev/next/autoplay buttons
            show_current_info: Whether to show current timestamp info
            hours_per_second: How many data hours should pass per real second during autoplay
            renders_per_second: How many times per second to update the display

        Returns:
            Tuple of (current_index, current_timestamp)
        """
        # Get timestamp data first - needed throughout the method
        timestamps, formatted_timestamps = self._get_timestamp_options()

        # Handle pending button actions
        button_action = st.session_state[f"{self.session_key}_button_action"]
        if button_action is not None:
            action, steps = button_action
            current_idx = self._get_current_index()

            if action == "next":
                new_idx = min(current_idx + steps, self.max_index)
            elif action == "prev":
                new_idx = max(current_idx - steps, self.min_index)
            else:
                new_idx = current_idx

            # Update the slider value directly for next render
            st.session_state[self.slider_key] = new_idx

            # Update session state immediately for external access
            new_timestamp = timestamps[new_idx]
            st.session_state[f"{self.session_key}_current_idx"] = new_idx
            st.session_state[f"{self.session_key}_current_timestamp"] = new_timestamp

            # Clear the action
            st.session_state[f"{self.session_key}_button_action"] = None

            # Stop autoplay if we hit boundaries
            if new_idx >= self.max_index or new_idx <= self.min_index:
                st.session_state[f"{self.session_key}_autoplay"] = False
                st.session_state[f"{self.session_key}_accum"] = 0.0

        # Autoplay: advance index based on accumulated time
        if self._is_autoplay_active() and len(self.data) >= 2:
            try:
                # Calculate time step between data points
                delta = self.data.index[1] - self.data.index[0]
                hours_per_step = delta.total_seconds() / 3600.0

                # Avoid division by zero and handle edge cases
                if hours_per_step > 0 and renders_per_second > 0:
                    hours_per_render = hours_per_second / renders_per_second
                    st.session_state[f"{self.session_key}_accum"] += hours_per_render
                    steps = int(
                        st.session_state[f"{self.session_key}_accum"] / hours_per_step
                    )
                    if steps > 0:
                        st.session_state[f"{self.session_key}_accum"] -= (
                            steps * hours_per_step
                        )
                        # Update slider state directly during autoplay
                        current_idx = self._get_current_index()
                        new_idx = min(current_idx + steps, self.max_index)
                        st.session_state[self.slider_key] = new_idx

                        # Update session state immediately for external access
                        new_timestamp = timestamps[new_idx]
                        st.session_state[f"{self.session_key}_current_idx"] = new_idx
                        st.session_state[f"{self.session_key}_current_timestamp"] = (
                            new_timestamp
                        )

                        if new_idx >= self.max_index:
                            # End of data reached, stop autoplay
                            st.session_state[f"{self.session_key}_autoplay"] = False
                            st.session_state[f"{self.session_key}_accum"] = 0.0
            except Exception as e:
                # If there's any error in autoplay calculation, stop autoplay gracefully
                st.error(f"Autoplay error: {str(e)}")
                st.session_state[f"{self.session_key}_autoplay"] = False

        # Render main slider; manual dragging stops autoplay
        selected_idx = st.select_slider(
            label,
            options=list(range(len(timestamps))),
            value=self._get_current_index(),
            format_func=lambda x: formatted_timestamps[x],
            key=self.slider_key,
            on_change=self._stop_autoplay,
        )

        # Use the current slider value as the source of truth
        current_idx = self._get_current_index()
        current_timestamp = timestamps[current_idx]

        # Update session state immediately when index changes
        st.session_state[f"{self.session_key}_current_idx"] = current_idx
        st.session_state[f"{self.session_key}_current_timestamp"] = current_timestamp

        # Show controls if requested
        if show_controls:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.button(
                    "⏮️ Prev",
                    disabled=current_idx <= self.min_index,
                    key=f"{self.session_key}_prev",
                    on_click=self._button_callback,
                    args=("prev", 1),
                )
            with col2:
                st.button(
                    "◀️◀️ -10",
                    disabled=current_idx <= self.min_index,
                    key=f"{self.session_key}_rewind",
                    on_click=self._button_callback,
                    args=("prev", 10),
                )
            with col3:
                st.button(
                    "⏭️ Next",
                    disabled=current_idx >= self.max_index,
                    key=f"{self.session_key}_next",
                    on_click=self._button_callback,
                    args=("next", 1),
                )
            with col4:
                st.button(
                    "⏭️⏭️ +10",
                    disabled=current_idx >= self.max_index,
                    key=f"{self.session_key}_fast_forward",
                    on_click=self._button_callback,
                    args=("next", 10),
                )
            with col5:
                autoplay_text = "⏸️ Stop" if self._is_autoplay_active() else "▶️ Play"
                # Use a regular button check instead of callback to avoid timing issues
                if st.button(autoplay_text, key=f"{self.session_key}_autoplay_btn"):
                    # Toggle autoplay immediately when button is clicked
                    current_state = st.session_state[f"{self.session_key}_autoplay"]
                    st.session_state[f"{self.session_key}_autoplay"] = not current_state

                    if not current_state:  # Starting autoplay
                        st.session_state[f"{self.session_key}_accum"] = 0.0
                    else:  # Stopping autoplay
                        st.session_state[f"{self.session_key}_accum"] = 0.0

                    # Don't force rerun here - let main.py handle it

        # Show current info and time scale information
        if show_current_info:
            if self._is_autoplay_active():
                # Add warning for large datasets during autoplay
                dataset_size_info = f" | **Dataset size:** {len(self.data)} points"
                if len(self.data) > 10000:
                    dataset_size_info += " ⚠️"
                st.info(
                    f"**Current Time:** {self._format_timestamp(current_timestamp)} | **Speed:** {hours_per_second}h/s | **Renders:** {renders_per_second}/s{dataset_size_info}"
                )
            else:
                st.info(
                    f"**Current Time:** {self._format_timestamp(current_timestamp)} | **Position:** {current_idx + 1}/{len(self.data)}"
                )

        # Handle autoplay rerun with improved timing
        current_autoplay_state = self._is_autoplay_active()

        # Return the final state - get fresh data to ensure it's current
        final_idx = self._get_current_index()
        final_timestamp = timestamps[final_idx]

        # Store the current state in a separate session variable for reliable access
        st.session_state[f"{self.session_key}_current_idx"] = final_idx
        st.session_state[f"{self.session_key}_current_timestamp"] = final_timestamp

        # For autoplay, we need to trigger rerun from main.py, not from here
        if current_autoplay_state:
            # Use a more efficient sleep mechanism
            sleep_time = 1.0 / renders_per_second
            if sleep_time > 0.05:  # Don't sleep for very short intervals
                time.sleep(sleep_time)
            # Don't call st.rerun() here - let main.py handle it

        # For button actions, we also don't rerun here
        # The button action has already been processed above

        return final_idx, final_timestamp
