"""Playback logic for RIWWER ML Demo."""
import time
import streamlit as st


def auto_advance(avg_hours_per_step: float) -> bool:
    """Handle auto-advance timing when playing.
    
    Args:
        avg_hours_per_step: Average hours per data step.
    
    Returns:
        True if playback should continue, False if stopped.
    """
    if st.session_state.skip_autoadvance:
        st.session_state.skip_autoadvance = False
        return True
    
    # Time-based speed calculation
    current_time = time.time()
    elapsed_real_time = current_time - st.session_state.last_update_time
    st.session_state.last_update_time = current_time
    
    # Get speed (hours per second)
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
        if new_idx >= st.session_state.get("max_index", 0):
            st.session_state.current_idx = 0
            st.session_state.last_update_time = time.time()
            st.session_state.accumulated_time = 0.0
            st.toast("🔄 Looped back to the beginning!", icon="🔄")
        else:
            st.session_state.current_idx = new_idx
    
    return True


def handle_loop(max_index: int) -> bool:
    """Handle loop when reaching the end of the data.
    
    Args:
        max_index: Maximum index value for the data.
    
    Returns:
        True if looped, False otherwise.
    """
    if st.session_state.current_idx >= max_index:
        st.session_state.current_idx = 0
        st.session_state.last_update_time = time.time()
        st.session_state.accumulated_time = 0.0
        st.toast("🔄 Looped back to the beginning!", icon="🔄")
        return True
    return False