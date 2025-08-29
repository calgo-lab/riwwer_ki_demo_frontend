import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional


class RainfallBarChart:
    """
    A rainfall bar chart with adjustable history window and 12h future horizon.

    - Shows actual rainfall as bars for the historical window up to current time
    - Shows forecast rainfall as bars for both history and the 12h future window
    - Uses different colors for actual vs forecast
    - Y-axis fixed to [max(0, min_in_window), max_in_window] computed from both series in window
    - Start time adjustable via hours back; default 72h
    - Optionally render as two distinct panels (actual vs forecast) to avoid overlap
    """

    def __init__(self, key: str = "rainfall_bar_chart"):
        self.key = key

    def render(
        self,
        data: pd.DataFrame,
        current_timestamp: pd.Timestamp,
        rainfall_column: str,
        rainfall_forecast_column: str,
        default_history_hours: int = 72,
        future_hours: int = 12,
        height: int = 280,
        show_controls: bool = True,
        is_playing: bool = False,
    separate_panels: bool = True,
    ) -> None:
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            st.info("No data available for rainfall chart.")
            return

        current_ts = pd.Timestamp(current_timestamp)

        # Controls
        # Use persisted value when controls are hidden (e.g., during playing)
        history_hours = int(st.session_state.get(f"{self.key}_history_hours_value", int(default_history_hours)))
        if show_controls and not is_playing:
            cols = st.columns([1, 5])
            with cols[0]:
                history_hours = st.number_input(
                    "History (h)",
                    min_value=1,
                    max_value=7 * 24,
                    value=int(history_hours),
                    step=6,
                    key=f"{self.key}_history_hours_ctrl",
                    help="Hours back from current time to show as history",
                )
        # Persist current selection/value
        st.session_state[f"{self.key}_history_hours_value"] = int(history_hours)

        window_start = current_ts - pd.Timedelta(hours=int(history_hours))
        window_end = current_ts + pd.Timedelta(hours=int(future_hours))

        data_start = data.index.min()
        actual_window_start = max(window_start, data_start)

        # Slice data
        window_mask = (data.index >= actual_window_start) & (data.index <= window_end)
        window_df = data.loc[window_mask, [c for c in [rainfall_column, rainfall_forecast_column] if c in data.columns]].copy()

        if window_df.empty:
            st.info("No rainfall data in selected window.")
            return

        # Prepare series
        if rainfall_column not in window_df.columns:
            window_df[rainfall_column] = pd.NA
        if rainfall_forecast_column not in window_df.columns:
            window_df[rainfall_forecast_column] = pd.NA

        # Zero-out/NaN actual rainfall in the future portion (don't plot actual for future)
        future_mask = window_df.index > current_ts
        window_df.loc[future_mask, rainfall_column] = pd.NA

        # Compute y-domain from both series in window
        actual_vals = pd.to_numeric(window_df[rainfall_column], errors="coerce")
        fcst_vals = pd.to_numeric(window_df[rainfall_forecast_column], errors="coerce")

        combined = pd.concat([actual_vals, fcst_vals], axis=0).dropna()
        if combined.empty:
            y_min, y_max = 0.0, 1.0
        else:
            min_val = combined.min()
            max_val = combined.max()
            y_min = max(0.0, float(min_val))
            y_max = float(max_val)
            if y_max <= y_min:
                y_max = y_min + 1.0

        # Build tidy dataframe for Altair
        tidy = pd.DataFrame({
            "datetime": window_df.index,
            "actual": actual_vals.values,
            "forecast": fcst_vals.values,
        })

        # Melt to long format with type column
        plot_df = tidy.melt(id_vars=["datetime"], value_vars=["actual", "forecast"], var_name="type", value_name="value")

        # Build charts
        if separate_panels:
            per_height = max(160, int(height / 2))

            # Actual panel
            actual_df = pd.DataFrame({
                'datetime': window_df.index,
                'value': actual_vals.values
            })
            actual_base = alt.Chart(actual_df.dropna()).transform_filter(
                alt.FieldGTPredicate(field='value', gt=0) | alt.FieldEqualPredicate(field='value', equal=0)
            ).encode(
                x=alt.X('datetime:T', scale=alt.Scale(domain=[actual_window_start, window_end]), title='Time'),
                y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Rainfall (mm)')
            )
            actual_bar = actual_base.mark_bar(color='#1f77b4', opacity=0.9)

            # Forecast panel
            forecast_df = pd.DataFrame({
                'datetime': window_df.index,
                'value': fcst_vals.values
            })
            forecast_base = alt.Chart(forecast_df.dropna()).transform_filter(
                alt.FieldGTPredicate(field='value', gt=0) | alt.FieldEqualPredicate(field='value', equal=0)
            ).encode(
                x=alt.X('datetime:T', scale=alt.Scale(domain=[actual_window_start, window_end]), title='Time'),
                y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Rainfall (mm)')
            )
            forecast_bar = forecast_base.mark_bar(color='#ff7f0e', opacity=0.75)

            # Current time rules
            current_line_df = pd.DataFrame({'x': [current_ts], 'y': [y_min]})
            current_rule_actual = alt.Chart(current_line_df).mark_rule(color='red', strokeWidth=2, opacity=0.7).encode(x='x:T')
            current_rule_forecast = alt.Chart(current_line_df).mark_rule(color='red', strokeWidth=2, opacity=0.7).encode(x='x:T')

            actual_chart = (actual_bar + current_rule_actual).properties(title='Actual rainfall', height=per_height)
            forecast_chart = (forecast_bar + current_rule_forecast).properties(title='Forecast rainfall', height=per_height)

            chart = alt.vconcat(actual_chart, forecast_chart).resolve_scale(y='shared')
        else:
            # Overlay with legend
            base = alt.Chart(plot_df).transform_filter(
                alt.FieldGTPredicate(field="value", gt=0) | alt.FieldEqualPredicate(field="value", equal=0)
            ).encode(
                x=alt.X('datetime:T', scale=alt.Scale(domain=[actual_window_start, window_end]), title='Time'),
                y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Rainfall (mm)'),
                color=alt.Color('type:N', scale=alt.Scale(domain=['actual','forecast'], range=['#1f77b4','#ff7f0e']), title='Series'),
                tooltip=[
                    alt.Tooltip('datetime:T', title='Time'),
                    alt.Tooltip('type:N', title='Series'),
                    alt.Tooltip('value:Q', title='Value (mm)', format='.2f')
                ]
            )
            bars = base.mark_bar(opacity=0.8)

            current_line_data = pd.DataFrame({'x': [current_ts, current_ts], 'y': [y_min, y_max]})
            current_line = alt.Chart(current_line_data).mark_rule(color='red', strokeWidth=2, opacity=0.7).encode(x='x:T')
            chart = (bars + current_line).properties(height=height)

        container = st.container()
        with container:
            st.altair_chart(chart, use_container_width=True, theme=None)
