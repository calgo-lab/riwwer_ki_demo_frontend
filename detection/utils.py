import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    df = pd.read_csv("data/vierlinden_21_22_23_all.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def check_one_sensor_status(
    df: pd.DataFrame, sensor_name: str, now: pd.Timestamp, defined_interval: int = 60
) -> dict:

    sensor_df = df[["Datetime", sensor_name]]
    sensor_df = sensor_df[sensor_df["Datetime"] <= now]

    last_valid_index = sensor_df[sensor_name].last_valid_index()

    if last_valid_index is None:
        return {
            "Sensor": sensor_name,
            "Status": "OFFLINE",
            "Value": None,
            "LastValidDataTime": None,
            "MinutesSinceLastValidData": None,
        }

    last_seen_time = sensor_df.loc[last_valid_index, "Datetime"]
    time_diff = now - last_seen_time
    minutes_diff = int(time_diff.total_seconds() / 60)

    status = "ONLINE" if minutes_diff <= defined_interval else "OFFLINE"

    return {
        "Sensor": sensor_name,
        "Status": status,
        "Value": sensor_df.loc[last_valid_index, sensor_name],
        "LastValidDataTime": last_seen_time.strftime("%Y-%m-%d-%H:%M:%S"),
        "TimeSinceLastValidData": str(time_diff),
    }


def check_all_sensors_status(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    for sensor in df.columns[1:]:
        status_info = check_one_sensor_status(df, sensor, now)
        print(status_info)


# Define sensor categories
sensor_categories = {
    "WaterLevel": [
        "FLP_Hohenstand_Pumpensumpf_pval",
        "FLP_Hohenstand_Becken1_pval",
        "FLP_Hohenstand_Becken3_pval",
        "FLP_Hohenstand_Beckne2_pval",
        "Kaiserstr_Füllstand_SWS_pval",
        "Kaiserstr_Füllstand_RWS_pval",
        "Kreuzweg_Füllstand_Pumpensumpf_pval",
        "Verlindenhof_Füllstand_Pumpensumpf_pval",
        "PV_18_Fuellstand_RUEB_1_ival",
        "PV_19_Fuellstand_RUEB_2_ival",
        "PV_20_Fuellstand_RUEB_3_ival",
        "PV_25_Fuellstand_RRB_ival",
        "Herzog_Oberwasser_pval",
        "Herzog_Unterwasser_pval",
    ],
    "Flow": [
        "Herzog_Durchflußmenge_pval",
        "Herzog_Berechnete_Durchflussmenge_pval",
        "FLP_Durchfluss_SWP1_und_SWP2_pval",
        "PV_15_Entleerung_RUEB_ival",
        "PV_16_Regenueberlauf_Menge_ival",
    ],
    "Pump": [
        "FLP_P3_pval",
        "FLP_P4_pval",
        "FLP_P5_pval",
        "Kaiserstr_P1_pval",
        "Kaiserstr_P2_pval",
        "Kaiserstr_P3_pval",
        "Kaiserstr_P4_pval",
        "Kaiserstr_P5_pval",
        "Kaiserstr_P6_pval",
        "Kreuzweg_Pumpe_1_pval",
        "Kreuzweg_Pumpe_2_pval",
        "Verlindenhof_Pumpe_1_pval",
        "Verlindenhof_Pumpe_2_pval",
        "Verlindenhof_Pumpe_3_pval",
    ],
    "Valve": ["Herzog_Schieber_Position_pval"],
    "Rain": ["Niederschlag_mm"],
}


def detect_waterlevel_anomalies(
    df: pd.DataFrame,
    level_col: str,
    now: pd.Timestamp,
    rain_col: str = "Niederschlag_mm",
    time_col: str = "Datetime",
    window_minutes: int = 180,
    rain_threshold: float = 1.0,
    level_tolerance: float = 0.05,
    min_valid_ratio: float = 0.8,
) -> dict:

    window_start = now - pd.Timedelta(minutes=window_minutes)
    window_df = df[(df[time_col] >= window_start) & (df[time_col] <= now)].copy()

    # No data in the defined time window
    if window_df.empty:
        return {
            "Sensor": level_col,
            "Status": "OFFLINE",
            "Description": "No data available in time window.",
        }

    # Incomplete data in the defined time window
    total_points = len(window_df)
    valid_points = window_df[level_col].notna().sum()
    valid_ratio = valid_points / total_points if total_points > 0 else 0

    if valid_ratio < min_valid_ratio:
        return {
            "Sensor": level_col,
            "Status": "OFFLINE",
            "Description": f"Sensor offline too often ({valid_ratio*100:.1f}%)",
        }
    # Interpolate missing values
    window_df[level_col] = window_df[level_col].interpolate()

    # Calculate rainfall and water level variation
    rain_sum = window_df[rain_col].sum()
    level_variation = window_df[level_col].max() - window_df[level_col].min()

    # Determine status based on rainfall and water level variation
    if rain_sum >= rain_threshold and level_variation <= level_tolerance:
        status = "Anomaly-A"
        desc = "Rainfall occurred but water level remained flat."
    elif rain_sum < 0.1 and level_variation > level_tolerance:
        status = "Anomaly-B"
        desc = "No rain but water level unexpectedly increased."
    else:
        status = "Normal"
        desc = "Water level change matches rainfall."

    return {
        "Sensor": level_col,
        "WindowStart": window_start.strftime("%Y-%m-%d %H:%M:%S"),
        "WindowEnd": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Status": status,
        "RainSum": round(rain_sum, 2),
        "LevelVariation": round(level_variation, 3),
        "Description": desc,
    }


def check_overflow_status(
    df: pd.DataFrame,
    sensor_name: str,
    now: pd.Timestamp,
    defined_interval: int = 60,
    threshold: float = 0.95,
) -> dict:

    sensor_df = df[["Datetime", sensor_name]]
    sensor_df = sensor_df[sensor_df["Datetime"] <= now]

    last_valid_index = sensor_df[sensor_name].last_valid_index()

    if last_valid_index is None:
        return {
            "Sensor": sensor_name,
            "Status": "OFFLINE",
            "Value": None,
            "LastValidDataTime": None,
            "MinutesSinceLastValidData": None,
        }

    last_seen_time = sensor_df.loc[last_valid_index, "Datetime"]
    time_diff = now - last_seen_time
    minutes_diff = int(time_diff.total_seconds() / 60)

    status = "ONLINE" if minutes_diff <= defined_interval else "OFFLINE"
    value = sensor_df.loc[last_valid_index, sensor_name]

    overflow_status = "True" if value > 6.0 * threshold else "False"

    return {
        "Sensor": sensor_name,
        "Status": status,
        "Overflow": overflow_status,
        "Value": sensor_df.loc[last_valid_index, sensor_name],
        "LastValidDataTime": last_seen_time.strftime("%Y-%m-%d-%H:%M:%S"),
        "TimeSinceLastValidData": str(time_diff),
    }


def plot_overflow(sensor_df, sensor_name, threshold):
    """
    Plot the sensor values with overflow threshold.
    """
    plt.figure(figsize=(12, 4))
    sns.barplot(x="Datetime", y=sensor_name, data=sensor_df, color="skyblue")

    sensor_df = sensor_df.tail(20)

    plt.axhline(
        y=threshold * 6.0,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold*6.0}",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sensor Value")
    plt.title(f"{sensor_name} Values with Overflow Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("overflow_analysis.png")


def analyze_rain_before_overflow(
    df: pd.DataFrame,
    sensor_name: str,
    rain_col: str = "Niederschlag_mm",
    time_col: str = "Datetime",
    overflow_threshold: float = 5.7,
    rain_window_minutes: int = 180,
    rain_trigger_threshold: float = 1.0,
    now: pd.Timestamp = None,
):
    df = df[[time_col, sensor_name, rain_col]].copy()
    if now:
        df = df[df[time_col] <= now]
    df = df.dropna(subset=[sensor_name])

    df["Overflow"] = df[sensor_name] > overflow_threshold
    df = df.sort_values(time_col).reset_index(drop=True)

    overflow_events = df[df["Overflow"]]

    rain_sums = []
    for event_time in overflow_events[time_col]:
        window_start = event_time - pd.Timedelta(minutes=rain_window_minutes)
        rain_window = df[(df[time_col] >= window_start) & (df[time_col] < event_time)]
        rain_sum = rain_window[rain_col].sum()
        rain_sums.append(rain_sum)

    # 添加结果列
    overflow_events = overflow_events.copy()
    overflow_events["RainBeforeOverflow"] = rain_sums
    overflow_events["RainTriggered"] = (
        overflow_events["RainBeforeOverflow"] >= rain_trigger_threshold
    )

    # 统计结果
    total_overflows = len(overflow_events)
    rain_triggered = overflow_events["RainTriggered"].sum()
    avg_rain = (
        overflow_events["RainBeforeOverflow"].mean() if total_overflows > 0 else 0
    )

    print(f"Total overflow events: {total_overflows}")
    print(
        f"Overflow events triggered by rain (>{rain_trigger_threshold}mm in {rain_window_minutes}min): {rain_triggered}"
    )
    print(f"Avg rainfall before overflow: {avg_rain:.2f} mm")

    return overflow_events


def analyze_rain_before_overflow(
    df: pd.DataFrame,
    sensor_name: str,
    now: pd.Timestamp,
    rain_col: str = "Niederschlag_mm",
    time_col: str = "Datetime",
    overflow_threshold: float = 5.7,  # 6.0*0.95
    rain_window_minutes: int = 180,
    rain_trigger_threshold: float = 1.0,
):
    # Filter and prepare the DataFrame
    df = df[[time_col, sensor_name, rain_col]].copy()
    df = df[df[time_col] <= now]
    df[sensor_name] = df[sensor_name].interpolate(limit_direction="both")
    df[rain_col] = df[rain_col].interpolate(limit_direction="both")

    # Check for overflow conditions
    df["Overflow"] = df[sensor_name] > overflow_threshold
    df = df.sort_values(time_col).reset_index(drop=True)
    overflow_events = df[df["Overflow"]]

    # Calculate rainfall before each overflow event
    rain_sums = []
    for event_time in overflow_events[time_col]:
        window_start = event_time - pd.Timedelta(minutes=rain_window_minutes)
        rain_window = df[(df[time_col] >= window_start) & (df[time_col] < event_time)]
        rain_sum = rain_window[rain_col].sum()
        rain_sums.append(rain_sum)

    # Create a new DataFrame for overflow events with rain data
    overflow_events = overflow_events.copy()
    overflow_events["RainBeforeOverflow"] = rain_sums
    overflow_events["RainTriggered"] = (
        overflow_events["RainBeforeOverflow"] >= rain_trigger_threshold
    )

    # 统计结果
    total_overflows = len(overflow_events)
    total_records = len(df)
    overflow_percentage = (
        (total_overflows / total_records) * 100 if total_records > 0 else 0
    )
    rain_triggered = overflow_events["RainTriggered"].sum()
    avg_rain = (
        overflow_events["RainBeforeOverflow"].mean() if total_overflows > 0 else 0
    )

    print(
        f"Total overflow events: {total_overflows} out of {total_records}, which is {overflow_percentage:.2f}%"
    )
    print(
        f"Overflow events triggered by rain (>{rain_trigger_threshold}mm in {rain_window_minutes}min): {rain_triggered}"
    )
    print(f"Avg rainfall before overflow: {avg_rain:.2f} mm")

    return overflow_events
