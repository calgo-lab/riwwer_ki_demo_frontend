import folium
import streamlit as st
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go
from streamlit_folium import st_folium
from streamlit.errors import StreamlitDuplicateElementKey

from utils.config import MAP_DATA, read_data, TARGET_COLUMN
from utils.dynamic_map_data import build_dynamic_map_data_from_row, load_map_data
from components import TimeSliderLive, SimulationChart

# Debugging
if "debugpy_initialized" not in st.session_state:
    try:
        import debugpy
        
        if not debugpy.is_client_connected():
            debugpy.listen(("localhost", 5678))
            print("Debugger listening on port 5678...")
        st.session_state.debugpy_initialized = True
    except ImportError:
        print("Debugging not available. Install debugpy for remote debugging.")
        st.session_state.debugpy_initialized = True
    except Exception as e:
        print(f"Error initializing debugger: {e}")
        st.session_state.debugpy_initialized = True

st.set_page_config(
    page_title="RIWWER KI Demo - Live Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("RIWWER KI Demo - Live Dashboard")
st.write("High-performance real-time dashboard with integrated map and simulation chart views")

# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]
map_data = load_map_data()  # Load once at startup

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

show_map = st.checkbox(
    "Show Real-time Map", value=True, help="Display interactive map in the dashboard"
)

# Checkbox for showing the simulation chart
show_simulation_chart = st.checkbox(
    "Show simulation view (72h window)",
    value=True,
    key="show_sim_chart_advanced",
)

# Performance optimization option
show_detailed_content = st.checkbox(
    "Show detailed content (rainfall, sensor trends)",
    value=False,
    key="show_detailed_content",
    help="Disable this for better chart performance during playback"
)

st.info(
    "🔧 **Live Dashboard**: Integrated real-time dashboard with map and simulation chart"
)

# Initialize components
time_slider = TimeSliderLive(vierlinden_data, session_key="main_live_advanced")
simulation_chart = SimulationChart(
    key=f"simulation_chart_advanced_{TARGET_COLUMN}", interactive=True
)

# Pre-calculate map center and static template (optimization)
map_center_lat, map_center_lon = 0, 0
popup_template = None
if show_map:
    # Get initial map data to calculate center
    sample_map_data = build_dynamic_map_data_from_row(map_data, vierlinden_data.iloc[0], vierlinden_data.index[0])
    map_center_lat = (
        sample_map_data["latitude"].min() + sample_map_data["latitude"].max()
    ) / 2
    map_center_lon = (
        sample_map_data["longitude"].min() + sample_map_data["longitude"].max()
    ) / 2
    # Pre-load popup template to avoid file I/O in the loop
    with open("templates/popup.html") as f:
        popup_template = Template(f.read())

# Define custom content renderer for advanced mode
def advanced_content_renderer(
    idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int
):
    # Current data display at the top
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

    # Main content area: Map and Simulation Chart side-by-side
    st.subheader("🗺️ Live Map & Simulation View")
    
    # Create two columns for side-by-side layout
    map_col, chart_col = st.columns([1, 1], gap="medium")

    # Right column: Simulation Chart (render first for better performance)
    with chart_col:
        if show_chart and show_simulation_chart:
            st.markdown("**📈 Simulation Chart**")
            simulation_chart.render(
                data=vierlinden_data,
                current_timestamp=timestamp,
                current_value=current_value,
                target_column=TARGET_COLUMN,
                iteration=iteration,
                show_checkbox=False,  # Disable internal checkbox to avoid conflicts
            )
        else:
            st.info("Chart display is disabled. Enable it in the configuration above.")

    # Left column: Interactive Map (render second to avoid blocking chart)
    with map_col:
        if show_map:
            st.markdown("**📍 Real-time Map**")        
            
            # Build optimized dynamic map data using current row
            MAP_DATA_DYNAMIC = build_dynamic_map_data_from_row(map_data, data_row, timestamp)

            # Create base map using pre-calculated center
            base_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=14)
            
            # Create feature group for dynamic markers
            fg = folium.FeatureGroup(name="Sensor_Markers")

            for _, row in MAP_DATA_DYNAMIC.iterrows():
                lat = row["latitude"]
                lon = row["longitude"]
                location_name = row["info"]
                sensor_statuses = row["sensor_statuses"]
                sensor_status_list = [(s["Sensor"], s["Status"]) for s in sensor_statuses]

                # Determine marker color based on sensor availability
                active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
                total_sensors = len(sensor_statuses)
                
                if total_sensors == 0:
                    marker_color = "gray"
                    status_text = "No sensors"
                elif active_sensors == total_sensors:
                    marker_color = "green"
                    status_text = "All sensors active"
                elif active_sensors > 0:
                    marker_color = "orange"
                    status_text = f"{active_sensors}/{total_sensors} sensors active"
                else:
                    marker_color = "red"
                    status_text = "All sensors inactive"

                popup_html = popup_template.render(
                    INFO=location_name,
                    LAT=f"{lat:.2f}",
                    LON=f"{lon:.2f}",
                    SENSORS=sensor_status_list,
                ) if popup_template else ""

                # Add main marker with color coding
                marker = folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color=marker_color, icon="rss", prefix="fa"),
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{location_name}: {status_text}"
                )
                fg.add_child(marker)

                # Add label marker with optimized styling
                label_color = {
                    "green": "darkgreen",
                    "orange": "darkorange", 
                    "red": "darkred",
                    "gray": "gray"
                }.get(marker_color, "darkred")
                
                label_marker = folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="position: relative; transform: translate(-50%, -50px);
                        z-index: 1000; font-size: 11px; font-weight: bold; color: {label_color};
                        white-space: nowrap; background: rgba(255, 255, 255, 0.9);
                        padding: 1px 3px; border-radius: 2px; border: 1px solid {label_color};">
                            {location_name}
                        </div>
                        """,
                        icon_size=(1, 1)  # Optimize icon size
                    ),
                )
                fg.add_child(label_marker)
            
            # Use dynamic st_folium with stable key but unique feature group
            st_folium(
                base_map,
                center=(map_center_lat, map_center_lon),
                zoom=14,
                feature_group_to_add=fg,
                width=600, 
                height=400,
                key="live_map_dynamic_advanced"
            )
            
            # Add compact legend for marker colors
            st.markdown("""
            **Legend:** 🟢 All active | 🟠 Partial | 🔴 Inactive | ⚫ No data
            """)
        else:
            st.info("Map display is disabled. Enable it in the configuration above.")

    # Additional content below (optional for performance)
    if show_detailed_content:
        st.markdown("---")
        
        # Regional Rainfall section
        st.subheader("🌧️ Regional Rainfall")
        
        # Filter data up to current timestamp
        map_data_filtered = map_data[map_data["Datetime"] <= timestamp]
        df_plot = map_data_filtered.set_index("Datetime")

        rainfall_cols = ["Niederschlag_mm", "Niederschlag_Vorhersage_mm"]
        available_cols = [col for col in rainfall_cols if col in df_plot.columns]

        if available_cols and not df_plot.empty:
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
            # Move the layout and chart rendering OUTSIDE the loop
            fig.update_layout(
                title="Rainfall Overview (Historical & Forecast)",
                xaxis_title="Time",
                yaxis_title="Rainfall (mm)",
                height=300,
                margin=dict(t=30, b=30),
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"rainfall_chart_advanced_{iteration}")

        # Sensor Trends by Location (only if map data is available)
        if show_map:
            st.subheader("📊 Sensor Trends by Location")

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
                            y_values = df_plot[sensor].fillna(0)

                            if len(y_values) > 0:
                                mean = y_values.mean()
                                std = y_values.std()
                                if std > 0:
                                    z_scores = (y_values - mean) / std
                                    anomalies = (z_scores.abs() > 3)
                                else:
                                    anomalies = pd.Series([False] * len(y_values), index=y_values.index)

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
                                fig.add_trace(
                                    go.Scatter(
                                        x=y_values.index,
                                        y=y_values,
                                        mode="lines",
                                        name=sensor,
                                        line=dict(color="blue"),
                                    )
                                )
                                if anomalies.any():
                                    fig.add_trace(
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
                                    fig, use_container_width=True, key=f"sensor_trend_advanced_{location}_{sensor}_{iteration}"
                                )
    else:
        st.info("💡 **Performance Mode**: Detailed content disabled for smoother playback. Enable above to see rainfall and sensor trends.")

    # Show current row data in expandable section (always available)
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

# Run live dashboard
time_slider.run_live_dashboard(
    content_renderer=advanced_content_renderer,
    hours_per_second=1.0,  # This will be overridden by speed controls
    updates_per_second=2.0,  # Increased update rate for smoother chart performance
    show_controls=True,
    show_progress=True,
    max_iterations=2000,
)