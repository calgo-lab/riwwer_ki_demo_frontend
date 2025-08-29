import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from jinja2 import Template

from utils.config import MAP_DATA, read_data, TARGET_COLUMN
from components import TimeSlider, SimulationChart

# Please use "streamlit run main.py" to run this app

st.title("RIWWER KI Demo Frontend")

st.write("This is a simple demo frontend for the RIWWER KI project.")
st.sidebar.header("Navigation")
st.sidebar.write("Use the sidebar to navigate through the app.")
st.sidebar.button("Home", on_click=lambda: st.write("Welcome to the Home page!"))
st.sidebar.button(
    "About", on_click=lambda: st.write("This app demonstrates the RIWWER KI project.")
)

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

vierlinden_data = read_data()[read_data().index >= "2023-01-01"]

st.dataframe(vierlinden_data)

# Initialize the time slider component
time_slider = TimeSlider(vierlinden_data, session_key="main_timeline")

# Render the time slider component
st.subheader("Time Navigation")
current_idx, current_timestamp = time_slider.render(
    label="Select Time Event",
    show_controls=True,
    show_current_info=True,
    hours_per_second=10.0,  # 10 hours pass per real second
    renders_per_second=2.0  # Update display 1 time per second
)

# Use the values directly returned from TimeSlider for immediate synchronization
current_row_data = vierlinden_data.iloc[current_idx]
current_value = current_row_data[TARGET_COLUMN]

# Display current data information
st.subheader("Current Data Point")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Current Index:** {current_idx}")
    st.write(f"**Current Timestamp:** {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.write(f"**{TARGET_COLUMN}:** {current_value:.2f}")
    st.write(f"**Position:** {current_idx + 1} of {len(vierlinden_data)}")

# Show full current row data
with st.expander("View Full Current Row Data"):
    st.dataframe(current_row_data)

# Add a visual indicator for the current position
st.write(f"📍 **Current Position:** {current_timestamp.strftime('%Y-%m-%d %H:%M')} - **{TARGET_COLUMN}:** {current_value:.2f}")

# Initialize and render the simulation chart component
simulation_chart = SimulationChart(key=f"simulation_chart_{TARGET_COLUMN}", interactive=True)
simulation_chart.render(
    data=vierlinden_data,
    current_timestamp=current_timestamp,
    current_value=current_value,
    target_column=TARGET_COLUMN
)

# Check if autoplay is active and trigger rerun from main.py
if st.session_state.get("main_timeline_autoplay", False):
    st.rerun()