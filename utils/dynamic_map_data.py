import pandas as pd
import streamlit as st
from utils.config import COORDINATES_DICT, SENSOR_GROUPS, DATA_PATH, CACHE_TIME, read_data


@st.cache_data(ttl=CACHE_TIME)
def load_map_data():
    """Load map data - reuses cached read_data() to avoid duplicate loading."""
    # Use the cached read_data() instead of loading the CSV again
    data = read_data()
    # Reset index for compatibility with existing code
    data = data.reset_index()
    return data


def get_sensor_status(
    df: pd.DataFrame, sensor_name: str, now: pd.Timestamp, defined_interval: int = 60
) -> dict:

    sensor_df = df[["Datetime", sensor_name]]
    sensor_df = sensor_df[sensor_df["Datetime"] <= now]

    last_valid_index = sensor_df[sensor_name].last_valid_index()

    if last_valid_index is None:
        return {
            "Sensor": sensor_name,
            "Status": "inactive",
            "Value": None,
            "LastValidDataTime": None,
            "MinutesSinceLastValidData": None,
        }

    last_seen_time = sensor_df.loc[last_valid_index, "Datetime"]
    time_diff = now - last_seen_time
    minutes_diff = int(time_diff.total_seconds() / 60)

    status = "active" if minutes_diff <= defined_interval else "inactive"

    return {
        "Sensor": sensor_name,
        "Status": status,
        "Value": sensor_df.loc[last_valid_index, sensor_name],
        "LastValidDataTime": last_seen_time.strftime("%Y-%m-%d-%H:%M:%S"),
        "TimeSinceLastValidData": str(time_diff),
    }


def build_dynamic_map_data(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for location, coords in COORDINATES_DICT.items():
        sensor_list = SENSOR_GROUPS[location]
        sensor_info = []
        for sensor in sensor_list:
            status_dict = get_sensor_status(df, sensor, now)
            sensor_info.append(status_dict)

        rows.append(
            {
                "latitude": coords[1],
                "longitude": coords[0],
                "info": location,
                "sensor_statuses": sensor_info,
            }
        )
    return pd.DataFrame(rows)


def build_dynamic_map_data_from_row(df: pd.DataFrame, current_row: pd.Series, now: pd.Timestamp) -> pd.DataFrame:
    """
    Optimized version that uses the current data row instead of reprocessing all data.
    This is more efficient for live dashboard updates.
    """
    rows = []
    for location, coords in COORDINATES_DICT.items():
        sensor_list = SENSOR_GROUPS[location]
        sensor_info = []
        for sensor in sensor_list:
            if sensor in current_row.index and pd.notna(current_row[sensor]):
                # Use current row data for active sensors
                status_dict = {
                    "Sensor": sensor,
                    "Status": "active",
                    "Value": current_row[sensor],
                    "LastValidDataTime": now.strftime("%Y-%m-%d-%H:%M:%S"),
                    "TimeSinceLastValidData": "0:00:00",
                }
            else:
                # Fall back to the original method for inactive sensors
                status_dict = get_sensor_status(df, sensor, now)
            sensor_info.append(status_dict)

        rows.append(
            {
                "latitude": coords[1],
                "longitude": coords[0],
                "info": location,
                "sensor_statuses": sensor_info,
            }
        )
    return pd.DataFrame(rows)
