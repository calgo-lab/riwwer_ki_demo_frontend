import streamlit as st
import pandas as pd
import time

# Please use "streamlit run main.py" to run this app

st.title("RIWWER KI Demo Frontend")

st.write("This is a simple demo frontend for the RIWWER KI project.")
st.sidebar.header("Navigation")
st.sidebar.write("Use the sidebar to navigate through the app.")
st.sidebar.button("Home", on_click=lambda: st.write("Welcome to the Home page!"))
st.sidebar.button(
    "About", on_click=lambda: st.write("This app demonstrates the RIWWER KI project.")
)

# Coordinates (long, lat)
coordinates_dict = {
    "sewage_treatment_facility": (6.7120366, 51.54740289999999),
    "kaiserstrasse": (6.707509099999999, 51.5402328),
    "kreuzweg": (6.710619200000001, 51.54230800000001),
    "vierlindenhof": (6.7371562, 51.5366435),
    "herzogstrasse": (6.723143199999999, 51.5433485),
    "franz_lenze_platz": (6.7233365, 51.5368246),
}

# Map data
map_data = pd.DataFrame(
    {
        "longitude": [coord[0] for coord in coordinates_dict.values()],
        "latitude": [coord[1] for coord in coordinates_dict.values()],
        "info": list(coordinates_dict.keys()),
    }
)
st.write("Map Data:")
st.dataframe(map_data)

st.subheader("Map of Locations")
st.map(map_data)

vierlinden_data = pd.read_csv("data/vierlinden_21_22_23_all_with_forecast.csv")

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
