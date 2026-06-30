"""Dashboard configuration panel for RIWWER ML Demo."""
import streamlit as st
from state.callbacks import on_scope_change_deferred


def render_dashboard_config_panel() -> None:
    """Render the dashboard configuration panel with model scope and type selectors."""
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
            on_change=on_scope_change_deferred,
            help=(
                "Standard operation: Uses all available sensor information within the network. "
                "Full network outage: Treats all external sensor information as inactive and use only local measurements for the forecasts."
            ),
        )
        
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