import os
import pandas as pd
import streamlit as st
import math

# Coordinates (long, lat)
COORDINATES_DICT = {
    "Kläranlage": (
        6.7120366,
        51.54740289999999,
    ),  # Sewage Treatment Facility
    "Kaiserstrasse": (6.707509099999999, 51.5402328),
    "Kreuzweg": (6.710619200000001, 51.54230800000001),
    "Vierlindenhof": (6.7371562, 51.5366435),
    "Herzogstrasse": (6.723143199999999, 51.5433485),
    "Franz Lenze Platz": (6.7233365, 51.5368246),
}
SENSOR_GROUPS = {
    "Kläranlage": [
        "Niederschlag_mm",
        "PV_15_Entleerung_RUEB_ival",
        # "PV_16_Regenueberlauf_Menge_ival", # defined overflow, exclude it
        "PV_18_Fuellstand_RUEB_1_ival",
        "PV_19_Fuellstand_RUEB_2_ival",
        "PV_20_Fuellstand_RUEB_3_ival",
        "PV_25_Fuellstand_RRB_ival",
    ],
    "Kaiserstrasse": [
        "Kaiserstr_Füllstand_SWS_pval",
        "Kaiserstr_Füllstand_RWS_pval",
        "Kaiserstr_P1_pval",
        "Kaiserstr_P2_pval",
        "Kaiserstr_P3_pval",
        "Kaiserstr_P4_pval",
        "Kaiserstr_P5_pval",
        "Kaiserstr_P6_pval",
    ],
    "Kreuzweg": [
        "Kreuzweg_Füllstand_Pumpensumpf_pval",
        "Kreuzweg_Pumpe_1_pval",
        "Kreuzweg_Pumpe_2_pval",
    ],
    "Vierlindenhof": [
        "Verlindenhof_Füllstand_Pumpensumpf_pval",
        "Verlindenhof_Pumpe_1_pval",
        "Verlindenhof_Pumpe_2_pval",
        "Verlindenhof_Pumpe_3_pval",
    ],
    "Herzogstrasse": [
        "Herzog_Schieber_Position_pval",
        "Herzog_Oberwasser_pval",
        "Herzog_Unterwasser_pval",
        "Herzog_Durchflußmenge_pval",
        # "Herzog_Berechnete_Durchflussmenge_pval", # theretically calculated, exclude it
    ],
    "Franz Lenze Platz": [
        "FLP_Hohenstand_Pumpensumpf_pval",
        "FLP_P3_pval",
        "FLP_P4_pval",
        "FLP_P5_pval",
        "FLP_Durchfluss_SWP1_und_SWP2_pval",
        "FLP_Hohenstand_Becken1_pval",
        "FLP_Hohenstand_Becken3_pval",
        "FLP_Hohenstand_Beckne2_pval",
    ],
}

MAP_DATA = pd.DataFrame(
    {
        "longitude": [coord[0] for coord in COORDINATES_DICT.values()],
        "latitude": [coord[1] for coord in COORDINATES_DICT.values()],
        "info": list(COORDINATES_DICT.keys()),
        "sensor_groups": [SENSOR_GROUPS[key] for key in COORDINATES_DICT.keys()],
    }
)

DATA_PATH = "data/vierlinden_21_22_23_all_with_forecast.csv"


@st.cache_data(ttl=900)
def read_data():
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=[0],
        index_col=0,
    )
    data.index.name = "Datetime"

    return data


def calculate_target_column_bounds(data: pd.DataFrame = None) -> tuple[float, float]:
    """
    Calculate fixed y-axis bounds for the target column.
    Returns min and max values rounded to the nearest ±0.5.

    Args:
        data: Optional DataFrame. If not provided, will load data using read_data()

    Returns:
        tuple: (y_min, y_max) rounded to nearest ±0.5
    """
    if data is None:
        data = read_data()

    if TARGET_COLUMN not in data.columns:
        # Fallback bounds if target column doesn't exist
        return -1.0, 1.0

    # Get min/max values, excluding NaN
    target_series = data[TARGET_COLUMN].dropna()
    if target_series.empty:
        # Fallback bounds if no valid data
        return -1.0, 1.0

    raw_min = target_series.min()
    raw_max = target_series.max()

    # Round to nearest 0.5
    # For min: round down (floor) to nearest 0.5
    # For max: round up (ceil) to nearest 0.5
    y_min = math.floor(raw_min * 2) / 2  # Round down to nearest 0.5
    y_max = math.ceil(raw_max * 2) / 2   # Round up to nearest 0.5

    # Ensure there's at least 0.5 difference between min and max
    if y_max - y_min < 0.5:
        y_max = y_min + 0.5

    return y_min, y_max


TARGET_COLUMN = "PV_18_Fuellstand_RUEB_1_ival"

RAINFALL_COLUMN = "Niederschlag_mm"

RAINFALL_FORECAST_COLUMN = "Niederschlag_Vorhersage_mm"

# Local predictions configuration
LOCAL_PREDICTIONS_PATH = "data/demo_local_predictions.csv"
PREDICTIONS_TIME_COLUMN = "Datetime"
LOCAL_LSTM_PRED_COLUMN = "LSTM Predictions"
LOCAL_TRANSFORMER_PRED_COLUMN = "Transformer Predictions"

# Global predictions configuration (12-step ahead arrays per timestamp)
GLOBAL_PREDICTIONS_PATH = "data/demo_global_predictions.csv"
GLOBAL_TFT_PRED_COLUMN = "TFT Predictions"
GLOBAL_LSTM_PRED_COLUMN = "LSTM Predictions"

# Overflow probability predictions
OVERFLOW_CLS_PRED_COLUMN = "Overflow Class Predictions"
OVERFLOW_CLS_PRED_FILE_PATH = os.path.join("data", "cls_hourly_predictions.csv")
