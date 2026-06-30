"""Map rendering for RIWWER ML Demo."""
import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.config import COORDINATES_DICT, SENSOR_GROUPS
from utils.dynamic_map_data import build_dynamic_map_data_from_row


# Marker size configuration
BASE_MARKER_SIZE = 80
RING_SIZE_MULTIPLIER = 1.7
ICON_SIZE_MULTIPLIER = 0.25


def get_marker_color_and_size(sensor_statuses: list) -> list:
    """Determine marker color based on sensor status.
    
    Args:
        sensor_statuses: List of sensor status dicts.
    
    Returns:
        RGBA color list.
    """
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


def prepare_map_data(
    map_data: pd.DataFrame,
    data_row: pd.Series,
    timestamp: pd.Timestamp,
    model_scope: str,
) -> tuple:
    """Prepare enhanced marker data for Pydeck visualization.
    
    Args:
        map_data: Static map data DataFrame.
        data_row: Current data row.
        timestamp: Current timestamp.
        model_scope: "Standard operation" or "Full network outage".
    
    Returns:
        Tuple of 5 DataFrames: (marker_base, marker_ring, marker_icon, label, target_highlight)
    """
    # Build dynamic map data using current row
    dynamic_map_data = build_dynamic_map_data_from_row(map_data, data_row, timestamp)

    # Prepare data for multiple layer types
    marker_base_points = []
    marker_ring_points = []
    marker_icons = []
    label_points = []
    target_highlight_points = []

    for _, row in dynamic_map_data.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        location_name = row["info"]
        sensor_statuses = row["sensor_statuses"]
        is_target_location = (location_name == "Kläranlage")

        # In Local mode, force all sensors to inactive for each location
        if model_scope == "Full network outage":
            sensor_list = SENSOR_GROUPS.get(location_name, [])
            sensor_statuses = [
                {
                    "Sensor": s,
                    "Status": "inactive",
                    "Value": None,
                    "LastValidDataTime": None,
                    "TimeSinceLastValidData": None,
                }
                for s in sensor_list
            ]

        # Get color based on sensor status
        color = get_marker_color_and_size(sensor_statuses)

        # Calculate status metrics
        active_sensors = sum(1 for s in sensor_statuses if s["Status"] == "active")
        total_sensors = len(sensor_statuses)
        activity_percentage = 100 * (active_sensors / max(total_sensors, 1))

        # Use consistent sizes for all markers
        base_size = BASE_MARKER_SIZE
        ring_size = BASE_MARKER_SIZE * RING_SIZE_MULTIPLIER
        icon_size = BASE_MARKER_SIZE * ICON_SIZE_MULTIPLIER

        # 1. Base marker (filled circle)
        marker_base_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': color,
            'size': base_size,
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 0,
            'note': 'Sewage Treatment Facility<br>Location of filling level forecasts' if is_target_location else ''
        })

        # 2. Outer ring for emphasis
        ring_color = color[:3] + [80]
        marker_ring_points.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': ring_color,
            'size': ring_size,
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 5
        })

        # 3. Icon center
        marker_icons.append({
            'lon': lon,
            'lat': lat,
            'location': location_name,
            'color': [156, 39, 176, 255] if is_target_location else [255, 255, 255, 255],
            'size': icon_size,
            'active_sensors': active_sensors,
            'inactive_sensors': total_sensors - active_sensors,
            'total_sensors': total_sensors,
            'activity_percentage': activity_percentage,
            'elevation': 10
        })

        # 3b. Dedicated subtle highlight ring for Kläranlage
        if is_target_location:
            target_highlight_points.append({
                'lon': lon,
                'lat': lat,
                'location': location_name,
                'size': base_size * (RING_SIZE_MULTIPLIER + 0.2),
                'elevation': 8
            })

        # 4. Labels
        label_color = [255, 255, 255, 255] if activity_percentage > 0.5 else [0, 0, 0, 255]
        label_bg_color = [0, 0, 0, 180] if activity_percentage > 0.5 else [255, 255, 255, 200]

        label_points.append({
            'lon': lon,
            'lat': lat - 0.0008,
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
        pd.DataFrame(label_points),
        pd.DataFrame(target_highlight_points)
    )


def render_map(
    center_lat: float,
    center_lon: float,
    marker_base_data: pd.DataFrame,
    marker_ring_data: pd.DataFrame,
    marker_icon_data: pd.DataFrame,
    label_data: pd.DataFrame,
    target_highlight_data: pd.DataFrame,
) -> None:
    """Render the Pydeck map.
    
    Args:
        center_lat: Map center latitude.
        center_lon: Map center longitude.
        marker_base_data: Base marker DataFrame.
        marker_ring_data: Ring marker DataFrame.
        marker_icon_data: Icon marker DataFrame.
        label_data: Label DataFrame.
        target_highlight_data: Target highlight DataFrame.
    """
    layers = []

    # 1. Outer ring layer
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

    # 1b. Target highlight ring for Kläranlage
    if target_highlight_data is not None and not target_highlight_data.empty:
        target_layer = pdk.Layer(
            'ScatterplotLayer',
            target_highlight_data,
            get_position='[lon, lat]',
            get_color=[156, 39, 176, 0],
            get_radius='size',
            pickable=False,
            filled=False,
            stroked=True,
            line_width_min_pixels=5,
            line_width_max_pixels=5,
            get_line_color=[156, 39, 176, 245],
            elevation_scale=2,
            elevation_range=[0, 30],
            get_elevation='elevation'
        )
        layers.append(target_layer)

    # 2. Base marker layer
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

    # 3. Icon center layer
    icon_layer = pdk.Layer(
        'ScatterplotLayer',
        marker_icon_data,
        get_position='[lon, lat]',
        get_color=[255, 255, 255, 255],
        get_radius='size',
        pickable=False,
        elevation_scale=3,
        elevation_range=[0, 40],
        get_elevation='elevation'
    )
    layers.append(icon_layer)

    # 4. Text labels
    if label_data is not None and not label_data.empty:
        label_bg_layer = pdk.Layer(
            'TextLayer',
            label_data,
            get_position='[lon, lat]',
            get_text='text',
            get_color=[128, 128, 128, 128],
            get_size=13,
            get_alignment_baseline="'middle'",
            get_text_anchor="'middle'",
            font_weight='bold'
        )
        layers.append(label_bg_layer)

        text_layer = pdk.Layer(
            'TextLayer',
            label_data,
            get_position='[lon, lat]',
            get_text='text',
            get_color=[255, 255, 255, 255],
            get_size=11,
            get_alignment_baseline="'middle'",
            get_text_anchor="'middle'",
            font_weight='bold'
        )
        layers.append(text_layer)

        # Create deck with stable view state
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=13.25,
            pitch=0,
            bearing=0
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                'html': '''
                    <div style="background: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px;">
                        <b>📍 {location}</b><br/>
                        <span style="color: #4CAF50;">●</span> Active: {active_sensors}<br/>
                        <span style="color: #FF5722;">●</span> Inactive: {inactive_sensors}<br/>
                        <i>{note}</i>
                    </div>
                ''',
                'style': {
                    'backgroundColor': 'transparent',
                    'border': 'none'
                }
            }
        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
            key="enhanced_pydeck_map",
            height=420,
        )