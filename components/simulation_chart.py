import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import time
import numpy as np
import json
try:
    from bokeh.plotting import figure
    from bokeh.models import Span, BoxAnnotation, Legend, LegendItem, DatetimeTickFormatter, HoverTool, ColumnDataSource, Label
except Exception:
    figure = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from streamlit.components.v1 import html as st_html
except Exception:
    st_html = None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _prepare_window_data(data_hash: str, 
                        window_start_ts: int, 
                        current_ts_int: int,
                        target_column: str) -> tuple[pd.DataFrame, bool]:
    """
    Cache expensive data filtering and preparation.
    
    Args:
        data_hash: Hash of the data for cache invalidation
        window_start_ts: Window start as timestamp integer
        current_ts_int: Current timestamp as integer  
        target_column: Name of the target column
    
    Returns:
        tuple: (filtered_data, has_valid_data)
    """
    # This will be called with the actual data in the main method
    # Placeholder for cache structure
    return pd.DataFrame(), False


@st.cache_data(ttl=600)  # Cache for 10 minutes
def _calculate_chart_bounds(data_hash: str, 
                          target_column: str, 
                          y_axis_bounds: Optional[tuple[float, float]]) -> tuple[float, float, float]:
    """
    Cache expensive y-axis domain calculations.
    
    Returns:
        tuple: (y_min, y_max, y_padding)
    """
    # This will be populated in the main method
    return 0.0, 1.0, 0.1


class SimulationChart:
    """
    A reusable simulation chart component for Streamlit apps.
    
    Features:
    - 72-hour rolling window with 12-hour future margin
    - Shows only historical data up to current time (no future spoilers)
    - Visual indicators for current time, future window, and missing data
    - Proper handling of edge cases (beginning of dataset, no data)
    - Clean simulation-style presentation
    
    Example:
        ```python
        from components.simulation_chart import SimulationChart
        
        # Create and render chart
        chart = SimulationChart()
        chart.render(
            data=vierlinden_data,
            current_timestamp=current_timestamp,
            current_value=current_value,
            target_column=TARGET_COLUMN,
            title="Water Level Simulation"
        )
        ```
    """
    
    def __init__(self, key: str, interactive: bool = False, *, renderer: str = "bokeh", mpl_fixed_height: bool = True):
        """Initialize the SimulationChart component."""
        self.key = key
        self.interactive = interactive
        self.renderer = (renderer or "bokeh").lower()
        # When using Matplotlib, keep a fixed pixel height (not responsive)
        # by default to avoid odd scaling when the container width changes.
        self.mpl_fixed_height = bool(mpl_fixed_height)
        # Performance optimization: Cache frequently used values
        self._chart_cache = {}
        self._last_data_hash = None
        self._title_cache = {}
        
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash for the data to enable caching."""
        # Use data shape and a few sample values for efficient hashing
        return f"{len(data)}_{data.index.min()}_{data.index.max()}"
        
    def _prepare_optimized_data(self, 
                               data: pd.DataFrame,
                               current_timestamp: pd.Timestamp,
                               target_column: str,
                               window_start: pd.Timestamp,
                               actual_window_start: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
        """Optimized data preparation with caching."""
        data_hash = self._get_data_hash(data)
        
        # Check if we can reuse cached data
        cache_key = f"{data_hash}_{current_timestamp.timestamp()}_{target_column}"
        if cache_key in self._chart_cache:
            return self._chart_cache[cache_key]
        
        # Filter data efficiently
        current_ts = pd.Timestamp(current_timestamp)
        historical_mask = (data.index >= actual_window_start) & (data.index <= current_ts)
        historical_data = data.loc[historical_mask, [target_column]].copy()
        
        # Check if we have valid data
        has_valid_data = not historical_data.empty and not historical_data[target_column].isna().all()
        
        # Cache the result
        result = (historical_data, has_valid_data)
        self._chart_cache[cache_key] = result
        
        # Limit cache size to prevent memory issues
        if len(self._chart_cache) > 50:
            # Remove oldest entries
            oldest_keys = list(self._chart_cache.keys())[:25]
            for old_key in oldest_keys:
                del self._chart_cache[old_key]
                
        return result
        
    def _get_cached_bounds(self, 
                          data: pd.DataFrame,
                          historical_data: pd.DataFrame, 
                          current_value: float,
                          target_column: str,
                          y_axis_bounds: Optional[tuple[float, float]]) -> tuple[float, float, float]:
        """Get y-axis bounds with caching for performance."""
        
        if y_axis_bounds is not None:
            # Use fixed y-axis bounds for consistent scaling
            y_min, y_max = y_axis_bounds
            y_padding = (y_max - y_min) * 0.05  # Smaller padding since we have fixed bounds
            return y_min, y_max, y_padding
            
        # Create cache key
        data_hash = self._get_data_hash(data)
        bounds_cache_key = f"bounds_{data_hash}_{target_column}"
        
        if bounds_cache_key in self._chart_cache:
            return self._chart_cache[bounds_cache_key]
        
        # Calculate bounds
        if not historical_data.empty and not historical_data[target_column].isna().all():
            # Use historical data bounds
            y_min = historical_data[target_column].min()
            y_max = historical_data[target_column].max()
            y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        else:
            # Fallback to current value-based scaling
            y_padding = abs(current_value * 0.1) if current_value != 0 else 1
            y_min = current_value - y_padding
            y_max = current_value + y_padding
            
        result = (y_min, y_max, y_padding)
        self._chart_cache[bounds_cache_key] = result
        return result
        
    def _clear_stale_cache(self, current_data_hash: str) -> None:
        """Clear cache when data changes to prevent stale data issues."""
        if self._last_data_hash and self._last_data_hash != current_data_hash:
            # Data has changed, clear relevant caches
            self._chart_cache.clear()
            self._title_cache.clear()
        self._last_data_hash = current_data_hash

    def render(self,
               data: pd.DataFrame,
               current_timestamp: pd.Timestamp,
               current_value: float,
               target_column: str,
               title: Optional[str] = None,
               height: int = 600,
               show_checkbox: bool = True,
               checkbox_label: str = "Show simulation view (72h window)",
               iteration: Optional[int] = None,
               y_axis_bounds: Optional[tuple[float, float]] = None,
               is_local_mode: Optional[bool] = None,
               forecast_value: Optional[float] = None,
               forecast_series: Optional[list[float]] = None) -> None:
        """
        Render the simulation chart component.
        
        Args:
            data: DataFrame with DatetimeIndex containing the time series data
            current_timestamp: Current time point in the simulation
            current_value: Current value at the current timestamp
            target_column: Name of the column to plot
            title: Optional custom title for the chart
            height: Height of the chart in pixels
            show_checkbox: Whether to show a checkbox to toggle the chart
            checkbox_label: Label for the checkbox
            iteration: Optional iteration number for unique keys
            y_axis_bounds: Optional tuple (y_min, y_max) for fixed y-axis scaling
        """
        # Show checkbox if requested
        if show_checkbox:
            # Create simple stable checkbox key that doesn't change during the session
            # Use only the chart's base key to avoid duplication and ensure stability
            checkbox_key = f"{self.key}_checkbox"
            
            if not st.checkbox(checkbox_label, key=checkbox_key):
                return
        
        # OPTIMIZATION: Clear stale cache when data changes
        current_data_hash = self._get_data_hash(data)
        self._clear_stale_cache(current_data_hash)
        
        if self.renderer == "bokeh":
            if figure is None:
                st.warning("Bokeh not installed.")
            else:
                self._create_bokeh_chart(
                    data=data,
                    current_timestamp=current_timestamp,
                    current_value=current_value,
                    target_column=target_column,
                    title=title,
                    height=height,
                    iteration=iteration,
                    y_axis_bounds=y_axis_bounds,
                    is_local_mode_hint=is_local_mode,
                    forecast_value=forecast_value,
                    forecast_series=forecast_series
                )
        elif self.renderer == "matplotlib":
            if plt is None:
                st.warning("Matplotlib not available.")
            else:
                self._create_matplotlib_chart(
                    data=data,
                    current_timestamp=current_timestamp,
                    current_value=current_value,
                    target_column=target_column,
                    title=title,
                    height=height,
                    iteration=iteration,
                    y_axis_bounds=y_axis_bounds,
                    is_local_mode_hint=is_local_mode,
                    forecast_value=forecast_value,
                    forecast_series=forecast_series
                )
        elif self.renderer == "uplot":
            if st_html is None:
                st.warning("Streamlit HTML components not available. Falling back to Altair.")
                try:
                    self._create_altair_chart(
                        data=data,
                        current_timestamp=current_timestamp,
                        current_value=current_value,
                        target_column=target_column,
                        title=title,
                        height=height,
                        iteration=iteration,
                        y_axis_bounds=y_axis_bounds,
                        is_local_mode_hint=is_local_mode,
                        forecast_value=forecast_value,
                        forecast_series=forecast_series
                    )
                except ImportError:
                    self._create_fallback_chart(data, target_column, height)
            else:
                self._create_uplot_chart(
                    data=data,
                    current_timestamp=current_timestamp,
                    current_value=current_value,
                    target_column=target_column,
                    title=title,
                    height=height,
                    iteration=iteration,
                    y_axis_bounds=y_axis_bounds,
                    is_local_mode_hint=is_local_mode,
                    forecast_value=forecast_value,
                    forecast_series=forecast_series
                )
        else:
            # Default to Bokeh
            if figure is not None:
                self._create_bokeh_chart(
                    data=data,
                    current_timestamp=current_timestamp,
                    current_value=current_value,
                    target_column=target_column,
                    title=title,
                    height=height,
                    iteration=iteration,
                    y_axis_bounds=y_axis_bounds,
                    is_local_mode_hint=is_local_mode,
                    forecast_value=forecast_value,
                    forecast_series=forecast_series
                )
                
    def _create_bokeh_chart(self,
                             data: pd.DataFrame,
                             current_timestamp: pd.Timestamp,
                             current_value: float,
                             target_column: str,
                             title: Optional[str] = None,
                             height: int = 600,
                             iteration: Optional[int] = None,
                             y_axis_bounds: Optional[tuple[float, float]] = None,
                             is_local_mode_hint: Optional[bool] = None,
                             forecast_value: Optional[float] = None,
                             forecast_series: Optional[list[float]] = None) -> None:
        """Create the chart using Bokeh and render via st.bokeh_chart."""

        # Determine mode
        if is_local_mode_hint is not None:
            is_local_mode = bool(is_local_mode_hint)
            is_global_mode = not is_local_mode
        else:
            is_global_mode = forecast_series is not None and len(forecast_series) > 0
            is_local_mode = (not is_global_mode) and (forecast_value is not None)

        history_window_hours = 24 if is_local_mode else 72
        current_ts = pd.Timestamp(current_timestamp)
        window_start = current_ts - pd.Timedelta(hours=history_window_hours)
        window_end = current_ts + (pd.Timedelta(hours=12) if is_global_mode else pd.Timedelta(hours=1))

        # Extend displayed x-range by +1h in local mode so t+1 points aren't clamped at the right frame
        plot_x_end = window_end + (pd.Timedelta(hours=1) if is_local_mode else pd.Timedelta(0))
        
        data_start = data.index.min()
        actual_window_start = max(window_start, data_start)

        # Data slices
        hist_mask = (data.index >= actual_window_start) & (data.index <= current_ts)
        fut_mask = (data.index > current_ts) & (data.index <= window_end)
        hist = data.loc[hist_mask, [target_column]].dropna()
        fut = data.loc[fut_mask, [target_column]].dropna()

        # Y bounds
        if y_axis_bounds is not None:
            y_min, y_max = y_axis_bounds
        else:
            vals = []
            if not hist.empty:
                vals.extend(hist[target_column].astype(float).tolist())
            if is_global_mode and not fut.empty:
                vals.extend(fut[target_column].astype(float).tolist())
            if is_local_mode:
                # consider t+1 gt & forecast
                pos = data.index.searchsorted(current_ts, side='right')
                if pos < len(data.index):
                    t1 = data.index[pos]
                    v = data.loc[t1, target_column]
                    if pd.notna(v):
                        vals.append(float(v))
                if forecast_value is not None:
                    vals.append(float(forecast_value))
            if forecast_series:
                vals.extend([float(v) for v in forecast_series[:12]])
            if not vals:
                vals = [0.0, 1.0]
            y_min, y_max = float(min(vals)), float(max(vals))
            if y_min == y_max:
                y_min, y_max = y_min - 0.5, y_max + 0.5

        # Convert to ColumnDataSource
        def to_cds(df: pd.DataFrame) -> ColumnDataSource:
            if df is None or df.empty:
                return ColumnDataSource(dict(datetime=[], value=[]))
            return ColumnDataSource(dict(datetime=list(df.index.to_pydatetime()), value=df[target_column].astype(float).tolist()))

        hist_src = to_cds(hist)
        fut_src = to_cds(fut)

        # Global forecast series
        series_src = ColumnDataSource(dict(datetime=[], value=[]))
        if is_global_mode and forecast_series:
            steps = min(12, len(forecast_series))
            times = [current_ts + pd.Timedelta(hours=i) for i in range(1, steps + 1)]
            series_src = ColumnDataSource(dict(datetime=[t.to_pydatetime() for t in times], value=[float(v) for v in forecast_series[:steps]]))

        # Local t+1 points
        gt_local_src = ColumnDataSource(dict(datetime=[], value=[]))
        fc_local_src = ColumnDataSource(dict(datetime=[], value=[]))
        if is_local_mode:
            pos = data.index.searchsorted(current_ts, side='right')
            if pos < len(data.index):
                t1 = data.index[pos]
                gt_val = data.loc[t1, target_column]
                if pd.notna(gt_val):
                    gt_local_src = ColumnDataSource(dict(datetime=[t1.to_pydatetime()], value=[float(gt_val)]))
                if forecast_value is not None:
                    fc_local_src = ColumnDataSource(dict(datetime=[t1.to_pydatetime()], value=[float(forecast_value)]))

        # Theme-aware styling
        try:
            current_theme = st.context.theme.type
        except Exception:
            current_theme = 'light'
        is_dark = (current_theme == 'dark')

        bg_color = '#0e1117' if is_dark else 'white'
        grid_color = '#2c2f36' if is_dark else 'lightgray'
        axis_color = '#e0e0e0' if is_dark else '#000000'
        text_color = '#e0e0e0' if is_dark else '#000000'

        # Build figure
        p = figure(
            x_axis_type='datetime',
            height=height,
            sizing_mode='stretch_width',
            title=title or f"{target_column} - Simulation View ({history_window_hours}h History + {12 if is_global_mode else 1}h Forecast)",
            toolbar_location='above',
            tools='pan,wheel_zoom,box_zoom,reset,save,hover'
        )
        p.x_range.start = window_start.to_pydatetime()
        p.x_range.end = plot_x_end.to_pydatetime()
        p.y_range.start = y_min
        p.y_range.end = y_max

        # Apply theme
        p.background_fill_color = bg_color
        p.border_fill_color = bg_color
        p.outline_line_color = grid_color
        p.xgrid.grid_line_color = grid_color
        p.ygrid.grid_line_color = grid_color
        p.xaxis.axis_label_text_color = axis_color
        p.yaxis.axis_label_text_color = axis_color
        p.xaxis.major_label_text_color = axis_color
        p.yaxis.major_label_text_color = axis_color
        p.title.text_color = text_color
        # Axis labels
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = target_column

        p.xaxis.formatter = DatetimeTickFormatter(minutes="%Y-%m-%d %H:%M", hours="%Y-%m-%d %H:%M", days="%Y-%m-%d")

        # Shaded regions
        if actual_window_start > window_start:
            missing_box = BoxAnnotation(left=window_start.to_pydatetime(), right=actual_window_start.to_pydatetime(), fill_color='lightblue', fill_alpha=0.15)
            p.add_layout(missing_box)
        # Extend future shading over the additional right margin in local mode
        future_right = plot_x_end if is_local_mode else window_end
        future_box = BoxAnnotation(left=current_ts.to_pydatetime(), right=future_right.to_pydatetime(), fill_color='lightgray', fill_alpha=0.2)
        p.add_layout(future_box)

        # Lines and points
        r_hist = p.line('datetime', 'value', source=hist_src, line_width=3, color='#1f77b4', legend_label='Historical')
        r_fut = None
        if is_global_mode and (not fut.empty):
            r_fut = p.line('datetime', 'value', source=fut_src, line_width=3, color='#9ecae1', legend_label='True Future (unknown)')
        r_series = None
        if is_global_mode and not series_src.data.get('datetime') == []:
            r_series = p.line('datetime', 'value', source=series_src, line_width=3, color='#ff7f0e', legend_label='Forecast (global)')
            p.circle('datetime', 'value', source=series_src, size=8, color='#ff7f0e')
        # Local points
        r_gt_local = None
        r_fc_local = None
        if is_local_mode:
            if gt_local_src.data.get('datetime'):
                r_gt_local = p.circle('datetime', 'value', source=gt_local_src, size=10, color='#9ecae1', legend_label='GT (t+1)')
            if fc_local_src.data.get('datetime'):
                r_fc_local = p.circle('datetime', 'value', source=fc_local_src, size=10, color='#ff7f0e', legend_label='Forecast (t+1)')

        # NOW vertical line and point at present
        now_span = Span(location=current_ts.timestamp() * 1000, dimension='height', line_color='red', line_dash='dashed', line_width=3)
        p.add_layout(now_span)
        # Red point at current time/value on the history with proper field names for hover
        try:
            now_src = ColumnDataSource(dict(datetime=[current_ts.to_pydatetime()], value=[float(current_value)]))
            p.circle('datetime', 'value', source=now_src, size=10, color='red', line_color='darkred')
        except Exception:
            pass
        # Now label near the top within the plotting area, add small horizontal offset from the line
        p.add_layout(Label(x=current_ts.timestamp() * 1000,
                           y=y_max,
                           x_units='data', y_units='data',
                           text='Now', text_color='red', text_font_style='bold', text_font_size='12pt',
                           text_baseline='top', x_offset=6))

        # Increase fonts
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        p.xaxis.axis_label_text_font_size = '13pt'
        p.yaxis.axis_label_text_font_size = '13pt'

        # Hover tool configuration
        hover = p.select_one(HoverTool)
        if hover:
            hover.tooltips = [
                ("Time", "@datetime{%F %T}"),
                (target_column, "@value{0.00}")
            ]
            hover.formatters = {"@datetime": "datetime"}
            hover.mode = 'vline'

        # Legend below chart: build explicit legend and add below, hide in-plot legend
        try:
            from bokeh.models import Legend, LegendItem
            p.legend.visible = False

            # Dummy renderer for 'Now' dashed line sample
            r_now = p.line([window_start.to_pydatetime(), window_start.to_pydatetime()], [y_min, y_min],
                           line_color='red', line_dash='dashed', line_width=2, visible=True)

            # Ensure we have renderers for future/forecast to show in legend even if absent
            if r_fut is None:
                r_fut = p.line([window_start.to_pydatetime(), window_start.to_pydatetime()], [y_min, y_min],
                               line_color='#9ecae1', line_width=3, visible=False)
            if r_series is None and r_fc_local is None:
                # create a dummy forecast renderer
                r_series = p.line([window_start.to_pydatetime(), window_start.to_pydatetime()], [y_min, y_min],
                                  line_color='#ff7f0e', line_width=3, visible=False)

            items = [
                LegendItem(label='Historical', renderers=[r_hist]),
                LegendItem(label='Now', renderers=[r_now]),
                LegendItem(label='True Future (unknown)', renderers=[r_fut]),
                LegendItem(label='Forecast', renderers=[r_series if r_series is not None else r_fc_local]),
            ]
            legend = Legend(items=items, orientation='horizontal')
            legend.click_policy = 'hide'
            legend.label_text_color = text_color
            legend.label_text_font_size = '13pt'
            legend.spacing = 12
            try:
                legend.location = 'center'
            except Exception:
                pass
            legend.background_fill_alpha = 0.0
            p.add_layout(legend, 'below')
        except Exception:
            # Fallback to default legend config if add_layout fails
            p.legend.click_policy = 'hide'
            p.legend.location = 'bottom_left'
            p.legend.label_text_color = text_color
            p.legend.background_fill_alpha = 0.0

        # Stable key
        if self.interactive and iteration is not None:
            stable_iteration = iteration % 100
            chart_key = f"{self.key}_bokeh_{stable_iteration}"
        else:
            chart_key = f"{self.key}_bokeh_static"

        # Streamlit's bokeh_chart does not support a key parameter
        st.bokeh_chart(p, use_container_width=True)

    def _create_matplotlib_chart(self,
                                 data: pd.DataFrame,
                                 current_timestamp: pd.Timestamp,
                                 current_value: float,
                                 target_column: str,
                                 title: Optional[str] = None,
                                 height: int = 600,
                                 iteration: Optional[int] = None,
                                 y_axis_bounds: Optional[tuple[float, float]] = None,
                                 is_local_mode_hint: Optional[bool] = None,
                                 forecast_value: Optional[float] = None,
                                 forecast_series: Optional[list[float]] = None) -> None:
        """Create chart using Matplotlib and render via st.pyplot."""

        if is_local_mode_hint is not None:
            is_local_mode = bool(is_local_mode_hint)
            is_global_mode = not is_local_mode
        else:
            is_global_mode = forecast_series is not None and len(forecast_series) > 0
            is_local_mode = (not is_global_mode) and (forecast_value is not None)

        history_window_hours = 24 if is_local_mode else 72
        current_ts = pd.Timestamp(current_timestamp)
        window_start = current_ts - pd.Timedelta(hours=history_window_hours)
        window_end = current_ts + (pd.Timedelta(hours=12) if is_global_mode else pd.Timedelta(hours=1))

        data_start = data.index.min()
        actual_window_start = max(window_start, data_start)

        hist = data.loc[(data.index >= actual_window_start) & (data.index <= current_ts), [target_column]].dropna()
        fut = data.loc[(data.index > current_ts) & (data.index <= window_end), [target_column]].dropna()

        if y_axis_bounds is not None:
            y_min, y_max = y_axis_bounds
        else:
            vals = []
            if not hist.empty:
                vals.extend(hist[target_column].astype(float).tolist())
            if is_global_mode and not fut.empty:
                vals.extend(fut[target_column].astype(float).tolist())
            if is_local_mode:
                pos = data.index.searchsorted(current_ts, side='right')
                if pos < len(data.index):
                    t1 = data.index[pos]
                    v = data.loc[t1, target_column]
                    if pd.notna(v):
                        vals.append(float(v))
                if forecast_value is not None:
                    vals.append(float(forecast_value))
            if forecast_series:
                vals.extend([float(v) for v in forecast_series[:12]])
            if not vals:
                vals = [0.0, 1.0]
            y_min, y_max = float(min(vals)), float(max(vals))
            if y_min == y_max:
                y_min, y_max = y_min - 0.5, y_max + 0.5

        try:
            current_theme = st.context.theme.type
        except Exception:
            current_theme = 'light'
        is_dark = (current_theme == 'dark')
        bg_color = '#0e1117' if is_dark else 'white'
        grid_color = '#2c2f36' if is_dark else 'lightgray'
        text_color = '#e0e0e0' if is_dark else '#000000'

        # Main plot figure
        if self.mpl_fixed_height:
            # Fix pixel height by controlling inches via DPI; apply a slight scale to reduce perceived zoom
            dpi = 90
            effective_height_px = max(200, int(height * 0.7))
            fig, ax = plt.subplots(figsize=(12, max(4, effective_height_px / dpi)), dpi=dpi)
        else:
            # Responsive width; height scales with aspect ratio
            fig, ax = plt.subplots(figsize=(10, max(4, height/100)))
        # Ensure no auto layout engine expands margins unpredictably
        try:
            fig.set_layout_engine(None)
        except Exception:
            pass
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Shaded regions
        if actual_window_start > window_start:
            ax.axvspan(window_start, actual_window_start, color='lightblue', alpha=0.15, lw=0)
        # Future shading: extend by +1h in local mode so t+1 markers aren't at the frame edge
        shade_end = window_end + (pd.Timedelta(hours=1) if is_local_mode else pd.Timedelta(0))
        ax.axvspan(current_ts, shade_end, color='lightgray', alpha=0.2, lw=0)

        # Lines
        if not hist.empty:
            ax.plot(hist.index, hist[target_column].astype(float), color='#1f77b4', lw=2, label='Historical')
        if is_global_mode and not fut.empty:
            ax.plot(fut.index, fut[target_column].astype(float), color='#9ecae1', lw=2, label='True Future (unknown)')
        if is_global_mode and forecast_series:
            steps = min(12, len(forecast_series))
            times = [current_ts + pd.Timedelta(hours=i) for i in range(1, steps + 1)]
            ax.plot(times, [float(v) for v in forecast_series[:steps]], color='#ff7f0e', lw=2, marker='o', ms=4, label='Forecast (global)')

        # Local points
        if is_local_mode:
            pos = data.index.searchsorted(current_ts, side='right')
            if pos < len(data.index):
                t1 = data.index[pos]
                gt_val = data.loc[t1, target_column]
                if pd.notna(gt_val):
                    ax.scatter([t1], [float(gt_val)], color='#9ecae1', s=30, label='GT (t+1)')
                if forecast_value is not None:
                    ax.scatter([t1], [float(forecast_value)], color='#ff7f0e', s=30, label='Forecast (t+1)')

        # NOW line and label
        ax.axvline(current_ts, color='red', lw=2, ls='--', zorder=5)
        # Red point at current (history) value
        try:
            ax.scatter([current_ts], [float(current_value)], color='red', edgecolors='darkred', s=50, zorder=6)
        except Exception:
            pass
        ax.text(current_ts, y_max, 'NOW', color='red', fontsize=10, fontweight='bold', ha='left', va='bottom', zorder=6)

        # Axes formatting
        # In local mode, extend plotting area by +1h so t+1 points aren't clamped
        xlim_right = window_end + (pd.Timedelta(hours=1) if is_local_mode else pd.Timedelta(0))
        ax.set_xlim([window_start, xlim_right])
        ax.set_ylim([y_min, y_max])
        # Remove extra x-margin to avoid automatic padding on the right
        try:
            ax.margins(x=0)
        except Exception:
            pass
        ax.grid(True, color=grid_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.set_title(title or f"{target_column} - Simulation View ({history_window_hours}h History + {12 if is_global_mode else 1}h Forecast)", color=text_color, loc='left')
        ax.set_xlabel('Date', color=text_color)
        ax.set_ylabel(target_column, color=text_color)

        # Date-only ticks on x-axis
        try:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # Align edge tick labels to keep text inside the axes without changing margins
            try:
                fig.canvas.draw()
                xtick_labels = ax.get_xticklabels()
                xtick_locs = ax.get_xticks()
                n = len(xtick_labels)
                if n:
                    # Leftmost stays centered (as requested)
                    xtick_labels[0].set_horizontalalignment('center')
                    xtick_labels[0].set_clip_on(True)

                    # Middle labels centered
                    for t in xtick_labels[1:-1]:
                        t.set_horizontalalignment('center')
                        t.set_clip_on(True)

                    # Rightmost conditionally right-aligned if <5h to the right edge
                    if n > 1:
                        try:
                            right_num = xtick_locs[-1]
                            right_dt = pd.to_datetime(mdates.num2date(right_num)).tz_localize(None)
                            hours_to_right = window_end - right_dt
                            align_right = hours_to_right < pd.Timedelta(hours=4)
                        except Exception:
                            align_right = True
                        xtick_labels[-1].set_horizontalalignment('right' if align_right else 'center')
                        xtick_labels[-1].set_clip_on(True)
            except Exception:
                pass
        except Exception:
            pass

        # Keep a consistent outer margin (no dynamic expansion of the frame)
        try:
            fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12)
        except Exception:
            pass

        # Add legend within the same figure to keep element count stable
        from matplotlib.lines import Line2D
        legend_labels = ['Historical', 'Now', 'True Future (unknown)', 'Forecast']
        legend_handles = [
            Line2D([0], [0], color='#1f77b4', lw=2),
            Line2D([0], [0], color='red', lw=2, linestyle='--'),
            Line2D([0], [0], color='#9ecae1', lw=2),
            Line2D([0], [0], color='#ff7f0e', lw=2, marker='o', markersize=4)
        ]
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            ncol=4,
            frameon=False,
            handlelength=2.5,
            handletextpad=0.6,
            columnspacing=1.2,
            bbox_to_anchor=(0.5, -0.02),
            prop={"size": 11}
        )
        try:
            fig.subplots_adjust(bottom=0.16)
        except Exception:
            pass
        # Fill the available width to auto-adjust horizontally
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    def _add_visual_elements(self,
                           fig: go.Figure,
                           historical_data: pd.DataFrame,
                           current_ts: pd.Timestamp,
                           current_value: float,
                           target_column: str,
                           window_start: pd.Timestamp,
                           window_end: pd.Timestamp,
                           actual_window_start: pd.Timestamp,
                           y_axis_bounds: Optional[tuple[float, float]] = None) -> None:
        """Add visual elements like shapes and annotations to the chart."""
        future_start = current_ts
        future_end = window_end
        
        # Get y-axis range for shading
        if y_axis_bounds is not None:
            # Use fixed y-axis bounds for consistent scaling
            y_min, y_max = y_axis_bounds
            y_padding = (y_max - y_min) * 0.05  # Smaller padding since we have fixed bounds
            
            # Update the figure's y-axis to use fixed bounds
            fig.update_layout(
                yaxis=dict(
                    range=[y_min, y_max],
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )
        elif not historical_data.empty:
            # Fallback to dynamic scaling based on historical data
            y_min = historical_data[target_column].min()
            y_max = historical_data[target_column].max()
            y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1  # Handle constant values
        else:
            # Fallback to current value-based scaling
            y_padding = abs(current_value * 0.1) if current_value != 0 else 1
            y_min = current_value - y_padding
            y_max = current_value + y_padding
            
            # Update the figure's y-axis to use fallback bounds
            fig.update_layout(
                yaxis=dict(
                    range=[y_min, y_max],
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )

        # Add visual elements (common to all branches)
        
        # Add vertical line at current time using a shape
        fig.add_shape(
            type="line",
            x0=current_ts,
            y0=y_min - y_padding,
            x1=current_ts,
            y1=y_max + y_padding,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            ),
            opacity=0.7
        )
        
        # Add "NOW" annotation
        fig.add_annotation(
            x=current_ts,
            y=y_max + y_padding * 0.5,
            text="NOW",
            showarrow=False,
            font=dict(color="red", size=12, family="Arial Black"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )
        
        # Add shaded region for future area
        fig.add_shape(
            type="rect",
            x0=future_start,
            y0=y_min - y_padding,
            x1=future_end,
            y1=y_max + y_padding,
            fillcolor="lightgray",
            opacity=0.2,
            line_width=0,
            layer="below"
        )
        
        # Add annotation for future area
        fig.add_annotation(
            x=future_start + (future_end - future_start) / 2,
            y=y_max,
            text="Future Window",
            showarrow=False,
            font=dict(color="gray", size=10),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Add shaded region for missing historical data (if applicable)
        if actual_window_start > window_start:
            fig.add_shape(
                type="rect",
                x0=window_start,
                y0=y_min - y_padding,
                x1=actual_window_start,
                y1=y_max + y_padding,
                fillcolor="lightblue",
                opacity=0.1,
                line_width=0,
                layer="below"
            )
            
            # Add annotation for missing historical data
            missing_hours = (actual_window_start - window_start).total_seconds() / 3600
            fig.add_annotation(
                x=window_start + (actual_window_start - window_start) / 2,
                y=y_min,
                text=f"No Data Available<br>({missing_hours:.0f}h missing)",
                showarrow=False,
                font=dict(color="blue", size=9),
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=1,
                opacity=0.8
            )
    
    def _create_fallback_chart(self, 
                              data: pd.DataFrame, 
                              target_column: str, 
                              height: int) -> None:
        """Create a fallback chart when altair is not available or there's an error."""
        try:
            # Simple Altair fallback
            chart_data = data[target_column].reset_index()
            chart_data = chart_data.rename(columns={'index': 'datetime'})
            
            # Handle height for fallback chart
            fallback_properties = {
                'title': f'{target_column} - Simple View',
                'height': height
            }
            
            fallback_chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('datetime:T', title='Time'),
                y=alt.Y(f'{target_column}:Q', title=target_column)
            ).properties(**fallback_properties)
            
            st.altair_chart(fallback_chart, use_container_width=True)
        except:
            # Ultimate fallback to Streamlit's built-in chart
            chart_data = data[target_column].copy()
            # For Streamlit's line_chart, we can only pass height if it's an int
            if isinstance(height, int):
                st.line_chart(chart_data, use_container_width=True, height=height)
            else:
                st.line_chart(chart_data, use_container_width=True)
