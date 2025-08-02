import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from jinja2 import Template

from utils.config import MAP_DATA, read_data, TARGET_COLUMN
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

with tab1:
    st.subheader("Map of Locations")
    
    # Calculate the center of the bounding box for the map view
    center_lat = (MAP_DATA["latitude"].min() + MAP_DATA["latitude"].max()) / 2
    center_lon = (MAP_DATA["longitude"].min() + MAP_DATA["longitude"].max()) / 2
    
    # Create a folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Load the HTML template for the popup
    with open("templates/popup.html") as f:
        popup_template = Template(f.read())
    
    # Add markers with click-popups
    for index, row in MAP_DATA.iterrows():
        popup_html = popup_template.render(
            INFO=row["info"],
            LAT=row["latitude"],
            LON=row["longitude"],
            SENSORS=row["sensor_groups"],
        )
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=100,  # radius in meters
            color="#C81E00",
            fill=True,
            fill_color="#C81E00",
            fill_opacity=0.5,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)
    
    # Render the map in Streamlit
    st_folium(m, width=725)

with tab2:
    st.subheader("Live Time Series Dashboard")
    
    # Show data overview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(vierlinden_data))
        with col2:
            st.metric("Date Range", f"{vierlinden_data.index.min().strftime('%Y-%m-%d')}")
            st.caption(f"to {vierlinden_data.index.max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Target Column", TARGET_COLUMN)
        
        st.dataframe(vierlinden_data.head(), use_container_width=True)
    
    # Configuration options
    st.subheader("⚙️ Dashboard Configuration")
    
    show_chart = st.checkbox("Show Real-time Chart", value=True, help="Display charts in the dashboard")
    
    # Dashboard Mode Selection
    st.subheader("🎛️ Dashboard Mode")
    dashboard_mode = st.radio(
        "Choose dashboard mode:",
        ["Simple Mode", "Advanced Mode"],
        index=0,
        horizontal=True
    )
    
    if dashboard_mode == "Simple Mode":
        st.info("🚀 **Simple Mode**: Basic real-time display with automatic chart generation")
        
        # Initialize and run the live time slider
        time_slider = TimeSliderLive(vierlinden_data, session_key="main_live")
        
        # Run simple mode
        time_slider.run_simple_loop(
            target_column=TARGET_COLUMN,
            hours_per_second=1.0,  # This will be overridden by speed controls
            updates_per_second=1.0,  # Fixed update rate
            show_chart=show_chart,
            max_iterations=2000  # Run for a long time
        )
    
    else:  # Advanced Mode
        st.info("🔧 **Advanced Mode**: Custom content renderer with simulation chart integration")
        
        # Checkbox for showing the simulation chart (outside the loop to avoid conflicts)
        show_simulation_chart = st.checkbox("Show simulation view (72h window)", value=True, key="show_sim_chart_advanced")
        
        # Initialize components
        time_slider = TimeSliderLive(vierlinden_data, session_key="main_live_advanced")
        simulation_chart = SimulationChart(key=f"simulation_chart_{TARGET_COLUMN}", interactive=True)

        # Define custom content renderer for advanced mode
        def advanced_content_renderer(idx: int, timestamp: pd.Timestamp, data_row: pd.Series, iteration: int):
            # Current data display
            st.subheader("📊 Current Data Point")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Index", f"{idx + 1:,} / {len(vierlinden_data):,}")
            with col2:
                st.metric("Timestamp", timestamp.strftime('%H:%M:%S'))
                st.caption(timestamp.strftime('%Y-%m-%d'))
            with col3:
                current_value = data_row[TARGET_COLUMN]
                st.metric(TARGET_COLUMN, f"{current_value:.2f}")
            with col4:
                # Calculate progress percentage
                progress_pct = (idx / max(1, len(vierlinden_data) - 1)) * 100
                st.metric("Progress", f"{progress_pct:.1f}%")
            
            # Visual indicator
            st.write(f"📍 **Current Position:** {timestamp.strftime('%Y-%m-%d %H:%M')} - **{TARGET_COLUMN}:** {current_value:.2f}")
            
            # Show simulation chart
            if show_chart and show_simulation_chart:
                st.subheader("📈 Simulation Chart")
                simulation_chart.render(
                    data=vierlinden_data,
                    current_timestamp=timestamp,
                    current_value=current_value,
                    target_column=TARGET_COLUMN,
                    iteration=iteration,
                    show_checkbox=False  # Disable internal checkbox to avoid conflicts
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
                    numeric_cols = row_df.select_dtypes(include=['number']).shape[1]
                    st.write(f"{numeric_cols} numeric columns")
        
        # Run advanced mode
        time_slider.run_live_dashboard(
            content_renderer=advanced_content_renderer,
            hours_per_second=1.0,  # This will be overridden by speed controls
            updates_per_second=2.0,  # Fixed update rate
            show_controls=True,
            show_progress=True,
            max_iterations=2000
        )

# Add footer
st.markdown("---")
st.markdown("**🔥 Performance Benefits:**")
st.markdown("""
- ✅ No page refreshes (uses `st.empty()` placeholder)
- ✅ Smooth real-time updates
- ✅ Lower CPU usage compared to `st.rerun()`
- ✅ Better user experience with continuous playback
- ✅ Configurable update rates and speeds
""")
