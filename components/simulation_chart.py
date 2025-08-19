import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from typing import Optional
import time
import numpy as np


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
    
    def __init__(self, key: str, interactive: bool = False):
        """Initialize the SimulationChart component."""
        self.key = key
        self.interactive = interactive
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
            st.warning("Altair not available. Install altair for enhanced visualization: `pip install altair`")
            self._create_fallback_chart(data, target_column, height)

    def _create_altair_chart(self,
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
        """Create the main altair chart with all visual elements."""

        # Decide mode based on provided forecasts or explicit hint
        if is_local_mode_hint is not None:
            is_local_mode = bool(is_local_mode_hint)
            is_global_mode = not is_local_mode
        else:
            is_global_mode = forecast_series is not None and len(forecast_series) > 0
            is_local_mode = (not is_global_mode) and (forecast_value is not None)

        # History window depends on mode: 72h for global, 24h for local (input sequence length)
        history_window_hours = 24 if is_local_mode else 72

        # Calculate the window ending at current time
        current_ts = pd.Timestamp(current_timestamp)
        window_start = current_ts - pd.Timedelta(hours=history_window_hours)
        # Visual future window: 12h for global (to show all 12 steps), 2h for local (just spacing)
        window_end = current_ts + (pd.Timedelta(hours=12) if is_global_mode else pd.Timedelta(hours=2))

        # Handle edge case: if we're at the beginning of the dataset
        data_start = data.index.min()
        actual_window_start = max(window_start, data_start)

        # OPTIMIZED: Use cached data preparation
        historical_data, has_valid_data = self._prepare_optimized_data(
            data=data,
            current_timestamp=current_ts,
            target_column=target_column,
            window_start=window_start,
            actual_window_start=actual_window_start
        )

        # Title should reflect fixed horizons: 72+12 for global, 24+1 for local (forecast horizon),
        # regardless of visual future margin (2h)
        future_steps = 12 if is_global_mode else 1
        if title is None:
            title = f"{target_column} - Simulation View ({history_window_hours}h History + {future_steps}h Forecast)"

        # OPTIMIZED: Use cached bounds calculation
        y_min, y_max, y_padding = self._get_cached_bounds(
            data=data,
            historical_data=historical_data,
            current_value=current_value,
            target_column=target_column,
            y_axis_bounds=y_axis_bounds
        )

        # In local mode, ensure y-domain also covers the t+1 points (forecast and GT) when dynamic bounds are used
        if y_axis_bounds is None and is_local_mode:
            # Determine t+1 timestamp (next index in data after current_ts)
            t_plus_one = None
            try:
                # Use searchsorted to find next timestamp strictly greater than current_ts
                pos = data.index.searchsorted(current_ts, side='right')
                if pos < len(data.index):
                    t_plus_one = data.index[pos]
            except Exception:
                t_plus_one = None

            extra_vals = []
            if t_plus_one is not None:
                # Ground truth value at t+1 if available
                try:
                    if target_column in data.columns and t_plus_one in data.index:
                        gt_val = data.loc[t_plus_one, target_column]
                        if pd.notna(gt_val):
                            extra_vals.append(float(gt_val))
                except Exception:
                    pass
            if forecast_value is not None:
                try:
                    extra_vals.append(float(forecast_value))
                except Exception:
                    pass

            if extra_vals:
                y_min = min(y_min, min(extra_vals))
                y_max = max(y_max, max(extra_vals))
                rng = (y_max - y_min)
                y_padding = (rng * 0.1) if rng != 0 else 0.1

        y_domain = [y_min, y_max]

        # OPTIMIZED: Create a unified data structure for all chart elements
        chart_elements = []

        # 1. Background rectangles for different time zones
        rect_data = []

        # Missing historical data rectangle (if applicable)
        if actual_window_start > window_start:
            missing_hours = (actual_window_start - window_start).total_seconds() / 3600
            rect_data.append({
                'x': window_start,
                'x2': actual_window_start,
                'y': y_domain[0],
                'y2': y_domain[1],
                'type': 'missing',
                'label': f'No Data Available ({missing_hours:.0f}h missing)'
            })

        # Future window rectangle (visual margin: 12h global, 2h local)
        rect_data.append({
            'x': current_ts,
            'x2': window_end,
            'y': y_domain[0],
            'y2': y_domain[1],
            'type': 'future',
            'label': f"Future Window ({'12h' if is_global_mode else '2h'})"
        })

        if rect_data:
            rect_df = pd.DataFrame(rect_data)

            # Background rectangles as separate charts
            for _, rect_row in rect_df.iterrows():
                background_rect = alt.Chart(pd.DataFrame([rect_row])).mark_rect(
                    opacity=0.15,
                    color='lightblue' if rect_row['type'] == 'missing' else 'lightgray'
                ).encode(
                    x=alt.X('x:T', scale=alt.Scale(domain=[window_start, window_end])),
                    x2='x2:T',
                    y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain)),
                    y2='y2:Q',
                    tooltip=['label:N']
                )
                chart_elements.append(background_rect)

        # 2. Historical data line
        if has_valid_data and not historical_data.empty:
            hist_df = historical_data.reset_index()
            if hist_df.columns[0] != 'datetime':
                hist_df = hist_df.rename(columns={hist_df.columns[0]: 'datetime'})

            if target_column in hist_df.columns:
                hist_df_clean = hist_df[hist_df[target_column].notna()]
                if not hist_df_clean.empty:
                    historical_line = alt.Chart(hist_df_clean).mark_line(
                        color='#1f77b4', strokeWidth=2.5, interpolate='linear', strokeCap='round'
                    ).encode(
                        x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                        y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                        tooltip=[
                            alt.Tooltip('datetime:T', title='Time'),
                            alt.Tooltip(f'{target_column}:Q', title=target_column, format='.2f')
                        ]
                    )
                    chart_elements.append(historical_line)

        # 2b. Ground truth future — line for global, single point for local
        try:
            if is_global_mode:
                future_mask = (data.index > current_ts) & (data.index <= window_end)
                future_df = data.loc[future_mask, [target_column]].dropna().reset_index()
                if not future_df.empty:
                    if future_df.columns[0] != 'datetime':
                        future_df = future_df.rename(columns={future_df.columns[0]: 'datetime'})
                    future_line = alt.Chart(future_df).mark_line(
                        color='#9ecae1', strokeWidth=2, interpolate='linear', strokeCap='round'
                    ).encode(
                        x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                        y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                        tooltip=[
                            alt.Tooltip('datetime:T', title='Time'),
                            alt.Tooltip(f'{target_column}:Q', title=f'{target_column} (Future GT)', format='.2f')
                        ]
                    )
                    chart_elements.append(future_line)
            elif is_local_mode:
                # Plot only t+1 ground truth as a point
                t_plus_one = None
                try:
                    pos = data.index.searchsorted(current_ts, side='right')
                    if pos < len(data.index):
                        t_plus_one = data.index[pos]
                except Exception:
                    t_plus_one = None

                if t_plus_one is not None and t_plus_one in data.index:
                    gt_val = data.loc[t_plus_one, target_column]
                    if pd.notna(gt_val):
                        gt_point_df = pd.DataFrame({'datetime': [t_plus_one], target_column: [float(gt_val)]})
                        gt_point = alt.Chart(gt_point_df).mark_point(color='#9ecae1', size=80).encode(
                            x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                            y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                            tooltip=[
                                alt.Tooltip('datetime:T', title='t+1'),
                                alt.Tooltip(f'{target_column}:Q', title='Ground truth (t+1)', format='.2f')
                            ]
                        )
                        chart_elements.append(gt_point)
        except Exception:
            pass

        # 2c. Model forecast — constant line for global, single point at t+1 for local
        if forecast_value is not None:
            try:
                if is_global_mode:
                    forecast_df = pd.DataFrame({'datetime': [current_ts, window_end], target_column: [forecast_value, forecast_value]})
                    forecast_line = alt.Chart(forecast_df).mark_line(color='#ff7f0e', strokeWidth=2.5, interpolate='linear', strokeCap='round').encode(
                        x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                        y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                        tooltip=[alt.Tooltip('datetime:T', title='Time'), alt.Tooltip(f'{target_column}:Q', title='Model forecast', format='.2f')]
                    )
                    chart_elements.append(forecast_line)
                elif is_local_mode:
                    t_plus_one = None
                    try:
                        pos = data.index.searchsorted(current_ts, side='right')
                        if pos < len(data.index):
                            t_plus_one = data.index[pos]
                    except Exception:
                        t_plus_one = None
                    if t_plus_one is not None:
                        forecast_point_df = pd.DataFrame({'datetime': [t_plus_one], target_column: [float(forecast_value)]})
                        forecast_point = alt.Chart(forecast_point_df).mark_point(color='#ff7f0e', size=80).encode(
                            x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                            y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                            tooltip=[alt.Tooltip('datetime:T', title='t+1'), alt.Tooltip(f'{target_column}:Q', title='Forecast (t+1)', format='.2f')]
                        )
                        chart_elements.append(forecast_point)
            except Exception:
                pass

        # 2d. Global multi-step forecast series (t+1..t+12) — orange with markers
        if is_global_mode and forecast_series is not None and len(forecast_series) > 0:
            try:
                steps = min(12, len(forecast_series))
                times = [current_ts + pd.Timedelta(hours=i) for i in range(1, steps + 1)]
                series_df = pd.DataFrame({'datetime': times, target_column: forecast_series[:steps]})
                series_line = alt.Chart(series_df).mark_line(color='#ff7f0e', strokeWidth=2.5).encode(
                    x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                    y=alt.Y(f'{target_column}:Q', scale=alt.Scale(domain=y_domain), title=target_column),
                    tooltip=[alt.Tooltip('datetime:T', title='Time'), alt.Tooltip(f'{target_column}:Q', title='Global forecast', format='.2f')]
                )
                series_pts = alt.Chart(series_df).mark_point(color='#ff7f0e', size=50).encode(
                    x='datetime:T', y=f'{target_column}:Q'
                )
                chart_elements.append(series_line + series_pts)
            except Exception:
                pass

        # 3. Current time vertical line
        current_line_data = pd.DataFrame({'x': [current_ts, current_ts], 'y': y_domain})
        current_line = alt.Chart(current_line_data).mark_line(color='red', strokeWidth=2, strokeDash=[5, 5], opacity=0.7).encode(
            x=alt.X('x:T', scale=alt.Scale(domain=[window_start, window_end])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain))
        )
        chart_elements.append(current_line)

        # 4. Current time point
        current_point_data = pd.DataFrame({'datetime': [current_ts], 'value': [current_value], 'label': ['Current Time']})
        current_point = alt.Chart(current_point_data).mark_circle(color='red', size=150, stroke='darkred', strokeWidth=2).encode(
            x=alt.X('datetime:T', scale=alt.Scale(domain=[window_start, window_end])),
            y=alt.Y('value:Q', scale=alt.Scale(domain=y_domain)),
            tooltip=[alt.Tooltip('datetime:T', title='Current Time'), alt.Tooltip('value:Q', title='Current Value', format='.2f')]
        )
        chart_elements.append(current_point)

        # 5. "NOW" text annotation
        now_text_data = pd.DataFrame({'x': [current_ts], 'y': [y_domain[1] - y_padding * 0.2], 'text': ['NOW']})
        now_text = alt.Chart(now_text_data).mark_text(align='center', baseline='middle', dx=0, dy=-10, fontSize=12, fontWeight='bold', color='red').encode(
            x=alt.X('x:T', scale=alt.Scale(domain=[window_start, window_end])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain)),
            text='text:N'
        )
        chart_elements.append(now_text)

        # Combine all chart elements
        if len(chart_elements) > 0:
            try:
                combined_chart = alt.layer(*chart_elements).resolve_scale(x='shared', y='shared').properties(
                    width='container', height=height, title={"text": title, "fontSize": 14, "anchor": "start"}
                )
            except Exception as e:
                st.error(f"Chart layering failed: {e}")
                combined_chart = None
        else:
            empty_data = pd.DataFrame({'x': [window_start, window_end], 'y': y_domain})
            combined_chart = alt.Chart(empty_data).mark_point(opacity=0).encode(
                x=alt.X('x:T', scale=alt.Scale(domain=[window_start, window_end]), title='Time'),
                y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain), title=target_column)
            ).properties(width='container', height=height, title=f'{target_column} - No Data')

        # Stable chart key
        if self.interactive and iteration is not None:
            stable_iteration = iteration % 100
            chart_key = f"{self.key}_altair_stable_{stable_iteration}"
        else:
            chart_key = f"{self.key}_altair_static"

        if combined_chart is not None:
            st.altair_chart(combined_chart, use_container_width=True, key=chart_key)
        else:
            st.warning(f"Unable to render chart for {target_column}. Chart creation failed.")
    
    def _create_chart(self,
                     data: pd.DataFrame,
                     current_timestamp: pd.Timestamp,
                     current_value: float,
                     target_column: str,
                     title: Optional[str] = None,
                     height: int = 500,
                     iteration: Optional[int] = None,
                     y_axis_bounds: Optional[tuple[float, float]] = None,
                     forecast_value: Optional[float] = None,
                     forecast_series: Optional[list[float]] = None) -> None:
        """Create the main plotly chart."""
        fig = go.Figure()
        
        # Calculate the 72-hour window ending at current time
        current_ts = pd.Timestamp(current_timestamp)
        window_start = current_ts - pd.Timedelta(hours=72)
        window_end = current_ts + pd.Timedelta(hours=12)  # 12h margin on the right
        
        # Handle edge case: if we're at the beginning of the dataset
        data_start = data.index.min()
        actual_window_start = max(window_start, data_start)
        
        # Filter data for the available window up to current time (no future data)
        historical_mask = (data.index >= actual_window_start) & (data.index <= current_ts)
        historical_data = data[historical_mask]
        
        # Add the historical data line (only up to current time)
        if not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[target_column],
                mode='lines',
                name=f'{target_column} (Historical)',
                line=dict(color='blue', width=2)
            ))
        
        # Add current point with enhanced visibility
        fig.add_trace(go.Scatter(
            x=[current_ts],
            y=[current_value],
            mode='markers',
            name='Current Time',
            marker=dict(
                color='red', 
                size=12, 
                symbol='circle',
                line=dict(color='darkred', width=2)
            ),
            showlegend=True
        ))

        # Add ground truth future line if available within future window
        try:
            future_mask = (data.index > current_ts) & (data.index <= window_end)
            future_data = data.loc[future_mask]
            if not future_data.empty:
                fig.add_trace(go.Scatter(
                    x=future_data.index,
                    y=future_data[target_column],
                    mode='lines',
                    name=f'{target_column} (Future GT)',
                    line=dict(color='rgb(158, 202, 225)', width=2, dash='solid')
                ))
        except Exception:
            pass

        # Add model forecast (constant across future window) if provided
        if forecast_value is not None:
            fig.add_trace(go.Scatter(
                x=[current_ts, window_end],
                y=[forecast_value, forecast_value],
                mode='lines',
                name='Model forecast',
                line=dict(color='rgb(255, 127, 14)', width=3)
            ))

        # Add global 12-step forecast series if provided
        if forecast_series is not None and len(forecast_series) > 0:
            steps = min(12, len(forecast_series))
            times = [current_ts + pd.Timedelta(hours=i) for i in range(1, steps + 1)]
            fig.add_trace(go.Scatter(
                x=times,
                y=forecast_series[:steps],
                mode='lines+markers',
                name='Global forecast',
                line=dict(color='rgb(255, 127, 14)', width=3),
                marker=dict(color='rgb(255, 127, 14)', size=6)
            ))
        
        # Calculate actual hours of historical data available
        if not historical_data.empty:
            actual_hours = (current_ts - historical_data.index.min()).total_seconds() / 3600
            hours_text = f"({actual_hours:.0f}h History" if actual_hours < 72 else "(72h History"
        else:
            hours_text = "(No History"
        
        # Generate title
        if title is None:
            title = f"{target_column} - Simulation View {hours_text} + 12h Future Window)"
        
        # Set fixed time range for simulation effect
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=target_column,
            height=height,
            xaxis=dict(
                range=[window_start, window_end],  # Fixed window (even if no data at start)
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            # Add some styling for simulation feel
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add visual elements (shapes and annotations)
        self._add_visual_elements(
            fig=fig,
            historical_data=historical_data,
            current_ts=current_ts,
            current_value=current_value,
            target_column=target_column,
            window_start=window_start,
            window_end=window_end,
            actual_window_start=actual_window_start,
            y_axis_bounds=y_axis_bounds
        )
        
        if self.interactive:
            # Create a unique key based on iteration count if available, otherwise fallback to stable key
            if iteration is not None:
                key = f"{self.key}_iter_{iteration}"
            else:
                # Use base key to avoid hash collisions
                key = f"{self.key}_interactive"
        else:
            key = self.key
        
        # Render the chart
        st.plotly_chart(fig, use_container_width=True, key=key)

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
