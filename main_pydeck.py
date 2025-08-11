import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import plotly.graph_objs as go
import altair as alt
from utils.config import read_data, TARGET_COLUMN, MAP_DATA, COORDINATES_DICT, SENSOR_GROUPS
from utils.dynamic_map_data import build_dynamic_map_data_from_row, load_map_data
from components import TimeSliderLive, SimulationChart

st.set_page_config(
    page_title="RIWWER KI Demo - Pydeck Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("RIWWER KI Demo - Smooth Pydeck Dashboard")
st.write("High-performance real-time dashboard with smooth, non-flickering map updates using Pydeck")

# Load data
vierlinden_data = read_data()[read_data().index >= "2023-01-01"]
map_data = load_map_data() # Why not just use vierlinden_data directly?

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

# Configuration options
st.subheader("⚙️ Dashboard Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    show_chart = st.checkbox("Show Real-time Chart", value=True)
with col2:
    show_map = st.checkbox("Show Smooth Map", value=True)  
with col3:
    show_labels = st.checkbox("Show Location Labels", value=True)

# Initialize components using the smooth TimeSliderLive approach
time_slider = TimeSliderLive(vierlinden_data, session_key="pydeck_main")
simulation_chart = SimulationChart(key="pydeck_simulation_chart", interactive=True)

# Pre-calculate map center based on bounding box (not just coordinate average)
# Get min/max bounds of all locations
lats = [coord[1] for coord in COORDINATES_DICT.values()]
lons = [coord[0] for coord in COORDINATES_DICT.values()]

min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)

# Calculate true geographic center of bounding box
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# Optional: Calculate optimal zoom level based on bounding box size
lat_range = max_lat - min_lat
lon_range = max_lon - min_lon
max_range = max(lat_range, lon_range)

# ========== MARKER SIZE CONFIGURATION ==========
# Adjust these values to fine-tune marker appearance
BASE_MARKER_SIZE = 80        # Base size for all markers (consistent across locations)
RING_SIZE_MULTIPLIER = 1.7    # How much larger the outer ring should be
ICON_SIZE_MULTIPLIER = 0.25   # Size of the center white dot relative to base
# ===============================================

def get_marker_color_and_size(sensor_statuses):
    """Determine marker color based on sensor status - SIZE IS NOW CONSISTENT"""
    if not sensor_statuses:
        return [128, 128, 128, 180]  # Gray for no sensors
    
    active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
    total_sensors = len(sensor_statuses)
    
    if total_sensors == 0:
        return [128, 128, 128, 180]  # Gray
    elif active_sensors == total_sensors:
        return [0, 255, 0, 200]  # Green - all active
    elif active_sensors > 0:
        return [255, 165, 0, 200]  # Orange - partial
    else:
        return [255, 0, 0, 200]  # Red - all inactive

def prepare_map_data(data_row, timestamp):
    """Prepare enhanced marker data for Pydeck visualization"""
    # Build dynamic map data using current row
    dynamic_map_data = build_dynamic_map_data_from_row(map_data, data_row, timestamp)
    
    # Prepare data for multiple layer types
    marker_base_points = []  # Base circles
    marker_ring_points = []  # Outer rings for emphasis
    marker_icons = []        # Icon-like center points
    label_points = []        # Text labels
    
    for _, row in dynamic_map_data.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        location_name = row["info"]
        sensor_statuses = row["sensor_statuses"]
        
        # Get color based on sensor status (size is now consistent)
        color = get_marker_color_and_size(sensor_statuses)
        
        # Calculate status metrics
        active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
        total_sensors = len(sensor_statuses)
        activity_percentage = 100 * (active_sensors / max(total_sensors, 1))
        
        # Use consistent sizes for all markers
        base_size = BASE_MARKER_SIZE
        ring_size = BASE_MARKER_SIZE * RING_SIZE_MULTIPLIER
        icon_size = BASE_MARKER_SIZE * ICON_SIZE_MULTIPLIER
        
        # Create marker-like visualization with multiple layers
        
        # 1. Base marker (filled circle) - CONSISTENT SIZE
        marker_base_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': color,
            'size': base_size,  # Same for all locations
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 0
        })
        
        # 2. Outer ring for emphasis - CONSISTENT SIZE
        ring_color = color[:3] + [80]  # Same color but more transparent
        marker_ring_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': ring_color,
            'size': ring_size,  # Consistent ring size
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 5
        })
        
        # 3. Icon center - CONSISTENT SIZE
        marker_icons.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': [255, 255, 255, 255],  # White center
            'size': icon_size,  # Consistent icon size
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 10
        })
        
        # 4. Labels with better positioning
        if show_labels:
            # Determine label color based on activity
            label_color = [255, 255, 255, 255] if activity_percentage > 0.5 else [0, 0, 0, 255]
            label_bg_color = [0, 0, 0, 180] if activity_percentage > 0.5 else [255, 255, 255, 200]
            
            label_points.append({
                'lon': lon,
                'lat': lat - 0.0008,  # Position below marker
                'text': location_name,
                'size': 11,
                'color': label_color,
                'background_color': label_bg_color,
                'active_sensors': active_sensors,
                'inactive_sensors': total_sensors - active_sensors,
                'total_sensors': total_sensors
            })
    
    return (
        pd.DataFrame(marker_base_points), 
        pd.DataFrame(marker_ring_points),
        pd.DataFrame(marker_icons),
        pd.DataFrame(label_points) if show_labels else None
    )

def smooth_content_renderer(idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int):
    """Content renderer for smooth updates using the TimeSliderLive pattern"""
    
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

    # Main content area: Split into three columns for better layout
    st.subheader("🗺️ Live Map & Simulation View")
    
    # Create three columns: Map | Overview | Chart
    map_col, overview_col, chart_col = st.columns([1.4, 0.5, 2], gap="small")

    # Left column: Interactive Map with Pydeck (smooth updates)
    with map_col:
        if show_map:
            st.markdown("**📍 Real-time Sensor Map**")
            
            # Prepare enhanced map data with multiple layers
            marker_base_data, marker_ring_data, marker_icon_data, label_data = prepare_map_data(data_row, timestamp)
            
            # Create multiple layers for marker-like appearance
            layers = []
            
            # 1. Outer ring layer (rendered first, behind everything)
            ring_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_ring_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                pickable=False,
                stroked=True,
                stroke_width_min_pixels=1,
                stroke_width_max_pixels=2,
                get_line_color=[255, 255, 255, 160],
                elevation_scale=1,
                elevation_range=[0, 20],
                get_elevation='elevation'
            )
            layers.append(ring_layer)
            
            # 2. Base marker layer (main colored circle)
            marker_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_base_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                pickable=True,
                auto_highlight=True,
                stroked=True,
                stroke_width_min_pixels=2,
                stroke_width_max_pixels=3,
                get_line_color=[255, 255, 255, 200],
                elevation_scale=2,
                elevation_range=[0, 30],
                get_elevation='elevation'
            )
            layers.append(marker_layer)
            
            # 3. Icon center layer (white dot to simulate RSS icon)
            icon_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_icon_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='size',
                pickable=False,
                elevation_scale=3,
                elevation_range=[0, 40],
                get_elevation='elevation'
            )
            layers.append(icon_layer)
            
            # 4. Text labels with background effect
            if show_labels and label_data is not None and not label_data.empty:
                # Background layer for labels (slightly larger, darker)
                label_bg_layer = pdk.Layer(
                    'TextLayer',
                    label_data,
                    get_position='[lon, lat]',
                    get_text='text',
                    get_color='background_color',
                    get_size=13,  # Slightly larger for background effect
                    get_alignment_baseline="'middle'",
                    get_text_anchor="'middle'",
                    font_weight='bold'
                )
                layers.append(label_bg_layer)
                
                # Foreground text layer
                text_layer = pdk.Layer(
                    'TextLayer',
                    label_data,
                    get_position='[lon, lat]',
                    get_text='text',
                    get_color='color',
                    get_size=11,
                    get_alignment_baseline="'middle'",
                    get_text_anchor="'middle'",
                    font_weight='bold'
                )
                layers.append(text_layer)
            
            # Create deck with stable view state (no flicker!)
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=13.25,
                pitch=0,  # Slight 3D angle to show elevation
                bearing=0
            )
            
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip=True
            )
            
            # This is the key - using a unique key based on iteration prevents flickering
            st.pydeck_chart(deck, use_container_width=True, key=f"enhanced_pydeck_map_{iteration}")
        else:
            st.info("Map display is disabled. Enable it in the configuration above.")

    # Middle column: Current Overview & Legend
    with overview_col:
        st.markdown("**📊 Current Overview**")
        
        if show_map:
            # Get the map data for overview calculations
            marker_base_data, _, _, _ = prepare_map_data(data_row, timestamp)
            
            # Calculate overview statistics
            total_locations = len(marker_base_data)
            active_locations = len(marker_base_data[marker_base_data['activity_percentage'] == 100])
            partial_locations = len(marker_base_data[(marker_base_data['activity_percentage'] > 0) & (marker_base_data['activity_percentage'] < 100)])
            inactive_locations = len(marker_base_data[marker_base_data['activity_percentage'] == 0])
            
            # Display as metrics
            st.metric("📍 Total Locations", total_locations)
            st.metric("🟢 Fully Active", active_locations)
            st.metric("🟠 Partially Active", partial_locations) 
            st.metric("🔴 Inactive", inactive_locations)
            
            # Calculate overall network health
            if total_locations > 0:
                network_health = ((active_locations * 100) + (partial_locations * 50)) / total_locations
                st.markdown("---")
                st.metric("🎯 Network Health", f"{network_health:.1f}%")
        else:
            st.info("Enable map to see overview statistics")

    # Right column: Simulation Chart  
    with chart_col:
        if show_chart:
            st.markdown("**📈 Simulation Chart**")
            simulation_chart.render(
                data=vierlinden_data,
                current_timestamp=timestamp,
                current_value=current_value,
                target_column=TARGET_COLUMN,
                iteration=iteration,
                show_checkbox=False,  # Disable internal checkbox to avoid conflicts
                height=600
            )
        else:
            st.info("Chart display is disabled. Enable it in the configuration above.")

    # New row below the map: Sensor trends by location (half width)
    st.markdown("---")
    details_col, _ = st.columns([1, 1], gap="medium")
    with details_col:
        if show_map:
            st.subheader("📊 Sensor Trends by Location")
            # Filter data up to current timestamp
            try:
                map_data_filtered = map_data[map_data["Datetime"] <= timestamp]
                df_plot = map_data_filtered.set_index("Datetime")
            except Exception:
                df_plot = pd.DataFrame()

            dynamic_df = build_dynamic_map_data_from_row(map_data, data_row, timestamp)

            for _, row in dynamic_df.iterrows():
                location = row["info"]
                sensor_statuses = row.get("sensor_statuses", [])
                if not sensor_statuses or not isinstance(sensor_statuses[0], dict):
                    continue

                sensor_names = [s.get("Sensor") for s in sensor_statuses if isinstance(s, dict) and "Sensor" in s]

                with st.expander(f"📍 {location} — {len(sensor_names)} sensors", expanded=False):
                    base_key = f"sensor_trends_{location}"                    
                    widget_key = f"{base_key}_{iteration}"

                    # Only render charts on demand to reduce lag
                    render_charts = st.checkbox(
                        "Render charts for this location",
                        value=st.session_state.get(base_key, False),
                        key=widget_key,
                        help="Enable to render time-series charts for all sensors at this location"
                    )
                    st.session_state[base_key] = render_charts

                    if render_charts:
                        for sensor in sensor_names:
                            if df_plot is not None and not df_plot.empty and sensor in df_plot.columns:
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
                                        (s.get("Status") for s in sensor_statuses if s.get("Sensor") == sensor),
                                        "UNKNOWN",
                                    )
                                    status_icon = (
                                        "🟢" if sensor_status == "active" else ("🔴" if sensor_status == "inactive" else "❓")
                                    )

                                    # Build Altair chart for performance
                                    df_sensor = y_values.reset_index()
                                    df_sensor.columns = ["Datetime", "value"]
                                    df_sensor["anomaly"] = anomalies.values

                                    line = (
                                        alt.Chart(df_sensor)
                                        .mark_line(color="#1f77b4")
                                        .encode(
                                            x=alt.X("Datetime:T", title="Time"),
                                            y=alt.Y("value:Q", title="Sensor Value"),
                                            tooltip=[
                                                alt.Tooltip("Datetime:T", title="Time"),
                                                alt.Tooltip("value:Q", title="Value", format=".2f"),
                                            ],
                                        )
                                    )

                                    if df_sensor["anomaly"].any():
                                        anomalies_df = df_sensor[df_sensor["anomaly"]]
                                        points = (
                                            alt.Chart(anomalies_df)
                                            .mark_point(color="red", size=60, filled=True)
                                            .encode(
                                                x="Datetime:T",
                                                y="value:Q",
                                                tooltip=[
                                                    alt.Tooltip("Datetime:T", title="Anomaly Time"),
                                                    alt.Tooltip("value:Q", title="Anomaly Value", format=".2f"),
                                                ],
                                            )
                                        )
                                        chart = (line + points).properties(
                                            title=f"{status_icon} {sensor} @ {location}", height=300
                                        )
                                    else:
                                        chart = line.properties(
                                            title=f"{status_icon} {sensor} @ {location}", height=300
                                        )

                                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Enable map to see location details")

# Run the smooth live dashboard using TimeSliderLive
time_slider.run_live_dashboard(
    content_renderer=smooth_content_renderer,
    hours_per_second=1.0,  # This will be controlled by the TimeSliderLive speed controls
    updates_per_second=2.0,  # Smooth update rate
    show_controls=True,
    show_progress=True,
    max_iterations=2000,
)

# Performance info
st.sidebar.markdown("### 🚀 Performance Features")
st.sidebar.markdown("""
- **Pydeck GPU acceleration** for smooth rendering
- **TimeSliderLive pattern** prevents page flicker  
- **Optimized data preparation** with minimal recomputation
- **Smart layer management** for reduced flickering
- **Configurable animation speed** via native controls
""")

st.sidebar.markdown("### 🎮 Controls Guide")
st.sidebar.markdown("""
- **Play/Pause**: Auto-animate through time
- **Speed Dropdown**: Control animation speed (0.1x to 15x)
- **Forward/Back**: Step manually through time
- **Start/End**: Jump to beginning/end
- **Map**: GPU-accelerated with preserved viewport
""")
