import streamlit as st
import pandas as pd
from typing import Optional
from bokeh.plotting import figure
from bokeh.models import HoverTool, Span, Range1d, ColumnDataSource
from bokeh.layouts import column as bokeh_column


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

        # Fixed x-axis range: exactly [-72h, +12h] around current time
        x_start_fixed = (current_ts - pd.Timedelta(hours=72))
        x_end_fixed = (current_ts + pd.Timedelta(hours=12))

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

        # Theme-aware styling
        try:
            current_theme = st.context.theme.type
        except Exception:
            current_theme = 'light'
        is_dark = (current_theme == 'dark')

        # Match SimulationChart theme colors
        bg_color = '#0e1117' if is_dark else 'white'
        grid_color = '#2c2f36' if is_dark else 'lightgray'
        axis_color = '#e0e0e0' if is_dark else '#000000'
        text_color = '#e0e0e0' if is_dark else '#000000'

        # Determine bar width from median interval (fallback 1h)
        idx = window_df.index.sort_values()
        if len(idx) >= 2:
            diffs = pd.Series(idx[1:]).reset_index(drop=True) - pd.Series(idx[:-1]).reset_index(drop=True)
            try:
                median_delta = pd.to_timedelta(pd.Series(diffs).dt.total_seconds().median(), unit='s')
            except Exception:
                median_delta = pd.Timedelta(hours=1)
        else:
            median_delta = pd.Timedelta(hours=1)
        bar_width_ms = max(1, int(median_delta.total_seconds() * 1000 * 0.8))

        # Shared y-range
        y_range = Range1d(start=y_min, end=y_max)

        def make_common_figure(fig_height: int, title: Optional[str] = None):
            p = figure(
                x_axis_type='datetime',
                height=int(fig_height),
                toolbar_location='above',
                x_range=(x_start_fixed.to_pydatetime(), x_end_fixed.to_pydatetime()),
                y_range=y_range,
                title=title,
            )
            p.sizing_mode = 'stretch_width'
            p.background_fill_color = bg_color
            p.border_fill_color = bg_color
            p.outline_line_color = grid_color
            p.xgrid.grid_line_color = grid_color
            p.ygrid.grid_line_color = grid_color
            p.xaxis.axis_label = 'Date'
            p.yaxis.axis_label = 'Rainfall (mm)'
            p.xaxis.major_label_text_color = axis_color
            p.yaxis.major_label_text_color = axis_color
            p.xaxis.axis_label_text_color = axis_color
            p.yaxis.axis_label_text_color = axis_color
            if p.title:
                p.title.text_color = text_color
            # Date tick formatter for consistency
            try:
                from bokeh.models import DatetimeTickFormatter
                p.xaxis.formatter = DatetimeTickFormatter(minutes="%Y-%m-%d %H:%M", hours="%Y-%m-%d %H:%M", days="%Y-%m-%d")
            except Exception:
                pass
            # Font sizes consistent with SimulationChart
            try:
                if p.title:
                    p.title.text_font_size = '14pt'
                p.xaxis.major_label_text_font_size = '12pt'
                p.yaxis.major_label_text_font_size = '12pt'
                p.xaxis.axis_label_text_font_size = '13pt'
                p.yaxis.axis_label_text_font_size = '13pt'
            except Exception:
                pass
            p.grid.grid_line_alpha = 0.3
            p.xgrid.minor_grid_line_color = None
            p.ygrid.minor_grid_line_color = None
            return p

        # Data sources (filter value >= 0 and not NaN)
        actual_mask = actual_vals.notna() & (pd.to_numeric(actual_vals, errors='coerce') >= 0)
        fcst_mask = fcst_vals.notna() & (pd.to_numeric(fcst_vals, errors='coerce') >= 0)

        actual_source = ColumnDataSource({
            'x': window_df.index[actual_mask].to_pydatetime(),
            'y': pd.to_numeric(actual_vals[actual_mask], errors='coerce').astype(float),
            'series': ['actual'] * int(actual_mask.sum()),
        })
        forecast_source = ColumnDataSource({
            'x': window_df.index[fcst_mask].to_pydatetime(),
            'y': pd.to_numeric(fcst_vals[fcst_mask], errors='coerce').astype(float),
            'series': ['forecast'] * int(fcst_mask.sum()),
        })

        actual_color = '#1f77b4'
        forecast_color = '#ff7f0e'

        # Build charts
        if separate_panels:
            per_height = max(160, int(height / 2))

            p_actual = make_common_figure(per_height, title='Actual rainfall')
            p_forecast = make_common_figure(per_height, title='Forecast rainfall')

            # Bars
            p_actual.vbar(x='x', top='y', width=bar_width_ms, source=actual_source, fill_color=actual_color, line_color=actual_color, fill_alpha=0.9)
            p_forecast.vbar(x='x', top='y', width=bar_width_ms, source=forecast_source, fill_color=forecast_color, line_color=forecast_color, fill_alpha=0.75)

            # Current time vertical line
            curr_span_actual = Span(location=current_ts.timestamp() * 1000, dimension='height', line_color='red', line_dash='dashed', line_width=3, line_alpha=1.0)
            curr_span_forecast = Span(location=current_ts.timestamp() * 1000, dimension='height', line_color='red', line_dash='dashed', line_width=3, line_alpha=1.0)
            p_actual.add_layout(curr_span_actual)
            p_forecast.add_layout(curr_span_forecast)

            # Hover tools
            hover_a = HoverTool(tooltips=[
                ('Time', '@x{%F %T}'),
                ('Value (mm)', '@y{0.00}')
            ], formatters={'@x': 'datetime'}, mode='vline')
            hover_f = HoverTool(tooltips=[
                ('Time', '@x{%F %T}'),
                ('Value (mm)', '@y{0.00}')
            ], formatters={'@x': 'datetime'}, mode='vline')
            p_actual.add_tools(hover_a)
            p_forecast.add_tools(hover_f)

            layout = bokeh_column(p_actual, p_forecast)
        else:
            p = make_common_figure(height)

            # Bars for both series
            r_actual = p.vbar(x='x', top='y', width=bar_width_ms, source=actual_source, fill_color=actual_color, line_color=actual_color, fill_alpha=0.9, legend_label='actual')
            r_forecast = p.vbar(x='x', top='y', width=bar_width_ms, source=forecast_source, fill_color=forecast_color, line_color=forecast_color, fill_alpha=0.75, legend_label='forecast')

            # Current time vertical line
            curr_span = Span(location=current_ts.timestamp() * 1000, dimension='height', line_color='red', line_dash='dashed', line_width=3, line_alpha=1.0)
            p.add_layout(curr_span)

            # Hover tool
            hover = HoverTool(tooltips=[
                ('Time', '@x{%F %T}'),
                ('Series', '@series'),
                ('Value (mm)', '@y{0.00}')
            ], formatters={'@x': 'datetime'}, mode='vline', renderers=[r_actual, r_forecast])
            p.add_tools(hover)

            # Legend below chart, horizontal, consistent font
            try:
                from bokeh.models import Legend, LegendItem
                p.legend.visible = False
                items = [
                    LegendItem(label='Actual', renderers=[r_actual]),
                    LegendItem(label='Forecast', renderers=[r_forecast]),
                ]
                legend = Legend(items=items, orientation='horizontal')
                legend.click_policy = 'hide'
                legend.label_text_color = text_color
                legend.label_text_font_size = '13pt'
                legend.spacing = 12
                legend.background_fill_alpha = 0.0
                p.add_layout(legend, 'below')
            except Exception:
                p.legend.label_text_color = text_color
                p.legend.background_fill_alpha = 0.0
                p.legend.click_policy = 'hide'

            layout = p

        container = st.container()
        with container:
            st.bokeh_chart(layout, use_container_width=True)
