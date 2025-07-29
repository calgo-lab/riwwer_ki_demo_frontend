import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from jinja2 import Template

from utils.config import MAP_DATA, read_data, TARGET_COLUMN

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
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=10,
        color="#C81E00",
        fill=True,
        fill_color="#C81E00",
        fill_opacity=0.5,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(m)

# Render the map in Streamlit
st_folium(m, width=725)

vierlinden_data = read_data()
st.dataframe(vierlinden_data)

MAX_LINES = len(vierlinden_data)
MIN_LINES = 0

if "autoplay" not in st.session_state:
    st.session_state["autoplay"] = False


def next_line():
    if st.session_state["slider1"] < MAX_LINES:
        st.session_state["slider1"] += 1
    else:
        pass  # todo: warning end of slider reached
    return


def prev_line():
    if st.session_state["slider1"] > MIN_LINES:
        st.session_state["slider1"] -= 1
    else:
        pass  # todo: warning start of slider reached
    return


def autoplay_clicked():
    st.session_state["autoplay"] = not st.session_state["autoplay"]


# --- AUTOPLAY LOGIC ---
if st.session_state["autoplay"]:
    if st.session_state["slider1"] < MAX_LINES:
        st.session_state["slider1"] += 1
    else:
        st.session_state["autoplay"] = False

timeevent = st.slider(
    "time event", min_value=MIN_LINES, max_value=MAX_LINES, key="slider1"
)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("selected timeevent number:", timeevent)
with col2:
    button_prev = st.button("prev", on_click=prev_line, key="button_prev")
with col3:
    button_next = st.button("next", on_click=next_line, key="button_next")
with col4:
    button_autoplay = st.button(
        "autoplay", on_click=autoplay_clicked, key="autoplaybutton"
    )
    st.write(st.session_state["autoplay"])

st.dataframe(vierlinden_data.iloc[timeevent])
st.write(
    f"Filling level of rain basin for event: {vierlinden_data.iloc[timeevent]['PV_18_Fuellstand_RUEB_1_ival']}"
)

# Visualize filling level with an area chart
min_value = vierlinden_data["PV_18_Fuellstand_RUEB_1_ival"].min()
max_value = vierlinden_data["PV_18_Fuellstand_RUEB_1_ival"].max()

st.subheader("Filling Level Area Chart")
st.area_chart(
    vierlinden_data["PV_18_Fuellstand_RUEB_1_ival"],
    use_container_width=True,
    height=300,
)

if st.session_state["autoplay"]:
    st.rerun()
