"""Widget callbacks for RIWWER ML Demo."""
import streamlit as st


def on_scope_change() -> None:
    """Callback when model scope changes - sets flag to trigger rerun after event handling."""
    st.session_state._scope_changed = True


def on_scope_change_deferred() -> bool:
    """Check if scope changed and trigger deferred rerun.
    
    Returns:
        True if rerun was triggered, False otherwise.
    """
    if st.session_state.get("_scope_changed", False):
        st.session_state._scope_changed = False
        st.rerun(scope="app")
        return True
    return False