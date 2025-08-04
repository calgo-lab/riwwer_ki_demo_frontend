import folium
import streamlit as st
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go
from streamlit_folium import st_folium

from utils.config import MAP_DATA, read_data, TARGET_COLUMN
from utils.dynamic_map_data import build_dynamic_map_data, load_map_data
from components import TimeSliderLive, SimulationChart

# Please use "streamlit run main_live.py" to run this app

st.set_page_config(
    page_title="RIWWER KI Demo - Live Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("RIWWER KI Demo - Live Dashboard")
st.write("High-performance real-time dashboard using st.empty() placeholder pattern")

# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]

# Create two tabs - one for the map, one for the live dashboard
tab1, tab2 = st.tabs(["📍 Map View", "⚡ Live Dashboard"])

# Add markers with click-popups
with tab1:

    # -------------------------------------------------------
    st.subheader("Map of Locations")

    now = pd.Timestamp(
        "2021-08-01 01:30:10"
    )  # TODO: @Vipin, replace with time from slider
    map_data = load_map_data()
    MAP_DATA_DYNAMIC = build_dynamic_map_data(map_data, now)

    center_lat = (
        MAP_DATA_DYNAMIC["latitude"].min() + MAP_DATA_DYNAMIC["latitude"].max()
    ) / 2
    center_lon = (
        MAP_DATA_DYNAMIC["longitude"].min() + MAP_DATA_DYNAMIC["longitude"].max()
    ) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    with open("templates/popup.html") as f:
        popup_template = Template(f.read())

    for _, row in MAP_DATA_DYNAMIC.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        location_name = row["info"]
        sensor_statuses = [(s["Sensor"], s["Status"]) for s in row["sensor_statuses"]]

        popup_html = popup_template.render(
            INFO=location_name,
            LAT=f"{lat:.2f}",
            LON=f"{lon:.2f}",
            SENSORS=sensor_statuses,
        )

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="darkred", icon="rss", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="position: relative; transform: translate(-50%, -50px);
                z-index: 1000; font-size: 12px; font-weight: bold; color: darkred;
                white-space: nowrap; background: rgba(255, 255, 255, 0.6);
                padding: 0px 3px; border-radius: 3px;">
                    {location_name}
                </div>
                """
            ),
        ).add_to(m)

    st_folium(m, width=725, height=500)

    # -------------------------------------------------------
    map_data_filtered = map_data[map_data["Datetime"] <= now]
    df_plot = map_data_filtered.set_index("Datetime")

    # -------------------------------------------------------
    st.subheader("Regional Rainfall")

    rainfall_cols = ["Niederschlag_mm", "Niederschlag_Vorhersage_mm"]
    available_cols = [col for col in rainfall_cols if col in df_plot.columns]

    if available_cols:
        fig = go.Figure()
        for col in available_cols:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot[col].fillna(0),
                    mode="lines",
                    name=col,
                )
            )
        fig.update_layout(
            title="🌧️ Rainfall Overview (Historical & Forecast)",
            xaxis_title="Time",
            yaxis_title="Rainfall (mm)",
            height=300,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    # -------------------------------------------------------
    st.markdown(
        "<hr style='margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True
    )
    st.subheader("Sensor Trends by Location")

    for index, row in MAP_DATA_DYNAMIC.iterrows():
        location = row["info"]
        sensor_statuses = row["sensor_statuses"]

        if not sensor_statuses or not isinstance(sensor_statuses[0], dict):
            continue

        sensor_names = [s["Sensor"] for s in sensor_statuses]

        with st.expander(
            f"📍 {location} — {len(sensor_names)} sensors", expanded=False
        ):
            for sensor in sensor_names:
                if sensor in df_plot.columns:
                    y_values = df_plot[sensor]
                    y_values = df_plot[sensor].fillna(0)

                    mean = y_values.mean()
                    std = y_values.std()
                    z_scores = (y_values - mean) / std
                    anomalies = (
                        z_scores.abs() > 3
                    )  # threshold is bigger, so fewer anomalies

                    sensor_status = next(
                        (s["Status"] for s in sensor_statuses if s["Sensor"] == sensor),
                        "UNKNOWN",
                    )
                    status_icon = (
                        "🟢"
                        if sensor_status == "active"
                        else "🔴" if sensor_status == "inactive" else "❓"
                    )

                    fig = go.Figure()
                    fig.add_trace(  # normal
                        go.Scatter(
                            x=y_values.index,
                            y=y_values,
                            mode="lines",
                            name=sensor,
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(  # anomalies
                        go.Scatter(
                            x=y_values.index[anomalies],
                            y=y_values[anomalies],
                            mode="markers",
                            name="Anomalies",
                            marker=dict(color="red", size=8, symbol="circle"),
                        )
                    )

                    fig.update_layout(
                        title=f"{status_icon} {sensor} @ {location}",
                        xaxis_title="Time",
                        yaxis_title="Sensor Value",
                        height=300,
                        margin=dict(t=30, b=30),
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"{location}_{sensor}"
                    )

with tab2:
    st.subheader("Live Time Series Dashboard")

    # Show data overview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(vierlinden_data))
        with col2:
            st.metric(
                "Date Range", f"{vierlinden_data.index.min().strftime('%Y-%m-%d')}"
            )
            st.caption(f"to {vierlinden_data.index.max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Target Column", TARGET_COLUMN)

        st.dataframe(vierlinden_data.head(), use_container_width=True)

    # Configuration options
    st.subheader("⚙️ Dashboard Configuration")

    show_chart = st.checkbox(
        "Show Real-time Chart", value=True, help="Display charts in the dashboard"
    )

    # Dashboard Mode Selection
    st.subheader("🎛️ Dashboard Mode")
    dashboard_mode = st.radio(
        "Choose dashboard mode:",
        ["Simple Mode", "Advanced Mode"],
        index=0,
        horizontal=True,
    )

    if dashboard_mode == "Simple Mode":
        st.info(
            "🚀 **Simple Mode**: Basic real-time display with automatic chart generation"
        )

        # Initialize and run the live time slider
        time_slider = TimeSliderLive(vierlinden_data, session_key="main_live")

        # Run simple mode
        time_slider.run_simple_loop(
            target_column=TARGET_COLUMN,
            hours_per_second=1.0,  # This will be overridden by speed controls
            updates_per_second=1.0,  # Fixed update rate
            show_chart=show_chart,
            max_iterations=2000,  # Run for a long time
        )

    else:  # Advanced Mode
        st.info(
            "🔧 **Advanced Mode**: Custom content renderer with simulation chart integration"
        )

        # Checkbox for showing the simulation chart (outside the loop to avoid conflicts)
        show_simulation_chart = st.checkbox(
            "Show simulation view (72h window)",
            value=True,
            key="show_sim_chart_advanced",
        )

        # Initialize components
        time_slider = TimeSliderLive(vierlinden_data, session_key="main_live_advanced")
        simulation_chart = SimulationChart(
            key=f"simulation_chart_{TARGET_COLUMN}", interactive=True
        )

        # Define custom content renderer for advanced mode
        def advanced_content_renderer(
            idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int
        ):
            # Current data display
            st.subheader("📊 Current Data Point")

            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Index", f"{idx + 1:,} / {len(vierlinden_data):,}")
            with col2:
                st.metric("Timestamp", timestamp.strftime("%H:%M:%S"))
                st.caption(timestamp.strftime("%Y-%m-%d"))
            with col3:
                current_value = data_row[TARGET_COLUMN]
                st.metric(TARGET_COLUMN, f"{current_value:.2f}")
            with col4:
                # Calculate progress percentage
                progress_pct = (idx / max(1, len(vierlinden_data) - 1)) * 100
                st.metric("Progress", f"{progress_pct:.1f}%")

            # Visual indicator
            st.write(
                f"📍 **Current Position:** {timestamp.strftime('%Y-%m-%d %H:%M')} - **{TARGET_COLUMN}:** {current_value:.2f}"
            )

            # Show simulation chart
            if show_chart and show_simulation_chart:
                st.subheader("📈 Simulation Chart")
                simulation_chart.render(
                    data=vierlinden_data,
                    current_timestamp=timestamp,
                    current_value=current_value,
                    target_column=TARGET_COLUMN,
                    iteration=iteration,
                    show_checkbox=False,  # Disable internal checkbox to avoid conflicts
                )

            # Show current row data in expandable section
            with st.expander("🔍 View Full Current Row Data"):
                # Display as a nice formatted table
                row_df = data_row.to_frame().T
                st.dataframe(row_df, use_container_width=True)

                # Show some statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Non-null values:**")
                    st.write(f"{row_df.notna().sum().iloc[0]} / {len(row_df.columns)}")
                with col2:
                    st.write("**Data types:**")
                    numeric_cols = row_df.select_dtypes(include=["number"]).shape[1]
                    st.write(f"{numeric_cols} numeric columns")

        # Run advanced mode
        time_slider.run_live_dashboard(
            content_renderer=advanced_content_renderer,
            hours_per_second=1.0,  # This will be overridden by speed controls
            updates_per_second=2.0,  # Fixed update rate
            show_controls=True,
            show_progress=True,
            max_iterations=2000,
        )

# Add footer
st.markdown("---")
st.markdown("**🔥 Performance Benefits:**")
st.markdown(
    """
- ✅ No page refreshes (uses `st.empty()` placeholder)
- ✅ Smooth real-time updates
- ✅ Lower CPU usage compared to `st.rerun()`
- ✅ Better user experience with continuous playback
- ✅ Configurable update rates and speeds
"""
)
