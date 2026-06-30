"""Session state manager for RIWWER ML Demo."""
import time
import streamlit as st


class StateManager:
    """Manages all session state for the application."""
    
    @staticmethod
    def init() -> None:
        """Initialize all session state variables for time navigation."""
        # Core time navigation state
        if "current_idx" not in st.session_state:
            st.session_state.current_idx = 0
        
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
            
        if "current_update_time_interval" not in st.session_state:
            st.session_state.current_update_time_interval = None
        
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
        
        # Time slider versioning
        if "time_slider_ver" not in st.session_state:
            st.session_state.time_slider_ver = 0
        
        if "last_idx" not in st.session_state:
            st.session_state.last_idx = 1  # 1-based for slider
        
        # App initialization flags
        if "app_init_done" not in st.session_state:
            st.session_state.app_init_done = False
        
        if "first_step_fix_applied" not in st.session_state:
            st.session_state.first_step_fix_applied = False
    
    @staticmethod
    def reset_timing() -> None:
        """Reset timing accumulators used by play/auto-advance."""
        st.session_state.last_update_time = time.time()
        st.session_state.accumulated_time = 0.0
        st.session_state.skip_autoadvance = True
    
    @staticmethod
    def step_backward() -> None:
        """Step one index back (bounded at 0), mirroring Back button behavior."""
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
        StateManager.reset_timing()
    
    @staticmethod
    def step_forward(max_index: int) -> int:
        """Step one index forward (bounded at max_index).
        
        Returns:
            The new index value.
        """
        new_idx = min(st.session_state.current_idx + 1, max_index)
        st.session_state.current_idx = new_idx
        StateManager.reset_timing()
        return new_idx