"""Navigation controls for RIWWER ML Demo."""
import streamlit as st
from state.manager import StateManager


# Speed options for playback
SPEED_OPTIONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]


def render_navigation_controls(max_index: int, min_interval: float, max_interval: float) -> None:
    """Render the main navigation controls (Back, Play/Pause, Forward, Status, Speed).
    
    Args:
        max_index: Maximum index value for the data.
        min_interval: Minimum update interval for stability.
        max_interval: Maximum update interval.
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Back button
    with col1:
        st.write("")
        back_disabled = st.session_state.is_playing or st.session_state.current_idx <= 0
        if st.button("Back", key="nav_back", icon="⏪", use_container_width=True, disabled=back_disabled):
            StateManager.step_backward()
    
    # Play/Pause buttons
    with col2:
        st.write("")
        if st.session_state.is_playing:
            if st.button("**Pause**", key="nav_pause_btn", type="primary", icon="⏸️", use_container_width=True):
                st.session_state.is_playing = False
                st.session_state.current_update_time_interval = None
                st.rerun(scope="app")
        else:
            if st.button("**Play**", key="nav_play_btn", type="primary", icon="▶️", use_container_width=True):
                st.session_state.is_playing = True
                st.session_state.current_update_time_interval = st.session_state.get("next_update_time_interval", 1.0)
                st.session_state.last_update_time = st.session_state.get("last_update_time", 1.0)
                st.session_state.accumulated_time = 0.0
                st.session_state.skip_autoadvance = True
                st.rerun(scope="app")
    
    # Forward button
    with col3:
        st.write("")
        forward_disabled = st.session_state.is_playing or st.session_state.current_idx >= max_index
        if st.button("Forward", key="nav_forward", icon="⏩", use_container_width=True, disabled=forward_disabled):
            new_idx = StateManager.step_forward(max_index)
            if new_idx >= max_index:
                st.session_state.is_playing = False
    
    # Status display
    with col4:
        st.write("")
        st.write("")
        if st.session_state.is_playing:
            st.write("**Status:** ▶️ Playing")
        else:
            st.write("**Status:** ⏸️ Paused")
    
    # Speed selector
    with col5:
        multiplier = st.selectbox(
            "Speed",
            SPEED_OPTIONS,
            index=5,  # Default to 10x
            format_func=lambda x: f"{x}x",
            key="speed_multiplier",
        )
        
        st.session_state.next_update_time_interval = min(max_interval, max(1 / multiplier, min_interval))
        # Update current interval if changed while playing
        if st.session_state.get("current_update_time_interval", None) != st.session_state.next_update_time_interval and st.session_state.is_playing:
            st.session_state.current_update_time_interval = st.session_state.next_update_time_interval
            st.rerun(scope="app")