import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import time


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

    def render(self,
               data: pd.DataFrame,
               current_timestamp: pd.Timestamp,
               current_value: float,
               target_column: str,
               title: Optional[str] = None,
               height: int = 500,
               show_checkbox: bool = True,
               checkbox_label: str = "Show simulation view (72h window)",
               iteration: Optional[int] = None) -> None:
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
        """
        # Show checkbox if requested
        if show_checkbox:
            # Create simple stable checkbox key that doesn't change during the session
            # Use only the chart's base key to avoid duplication and ensure stability
            checkbox_key = f"{self.key}_checkbox"
            
            if not st.checkbox(checkbox_label, key=checkbox_key):
                return
        
        try:
            self._create_chart(
                data=data,
                current_timestamp=current_timestamp,
                current_value=current_value,
                target_column=target_column,
                title=title,
                height=height,
                iteration=iteration
            )
        except ImportError:
            st.warning("Plotly not available. Install plotly for enhanced visualization: `pip install plotly`")
            self._create_fallback_chart(data, target_column, height)
    
    def _create_chart(self,
                     data: pd.DataFrame,
                     current_timestamp: pd.Timestamp,
                     current_value: float,
                     target_column: str,
                     title: Optional[str] = None,
                     height: int = 500,
                     iteration: Optional[int] = None) -> None:
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
            actual_window_start=actual_window_start
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
                           actual_window_start: pd.Timestamp) -> None:
        """Add visual elements like shapes and annotations to the chart."""
        future_start = current_ts
        future_end = window_end
        
        # Get y-axis range for shading
        if not historical_data.empty:
            y_min = historical_data[target_column].min()
            y_max = historical_data[target_column].max()
            y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1  # Handle constant values
            
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
                text="Future Window<br>(No Data)",
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
        else:
            # If no historical data at all, create a basic y-range around current value
            y_padding = abs(current_value * 0.1) if current_value != 0 else 1
            fig.update_layout(
                yaxis=dict(
                    range=[current_value - y_padding, current_value + y_padding],
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )
    
    def _create_fallback_chart(self, 
                              data: pd.DataFrame, 
                              target_column: str, 
                              height: int) -> None:
        """Create a fallback chart when plotly is not available."""
        chart_data = data[target_column].copy()
        st.line_chart(chart_data, use_container_width=True, height=height)
