import pandas as pd

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


def read_data():
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=[0],
        index_col=0,
    )
    data.index.name = "Datetime"

    return data


TARGET_COLUMN = "PV_18_Fuellstand_RUEB_1_ival"
