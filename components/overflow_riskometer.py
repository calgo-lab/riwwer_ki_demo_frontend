import streamlit as st
import pandas as pd

class OverflowRiskometer:
    """
    A Streamlit component for displaying overflow riskometer at Verlinden Sewage Treatment Facility location.

    This class creates an interactive riskometer that visualizes the probability
    of overflow events based on HistGradientBoostingClassifier algorithm predictions.
    Note that the target variable here is the RegenüberlaufMenge column in WBD dataset.
    The risk is categorized into three levels: LOW, MEDIUM, and HIGH, with corresponding
    color-coded visual indicators.

    Attributes:
        cls_predictions (pd.DataFrame): DataFrame containing CLS prediction data
                                       with timestamps as index and 'predicitons_hourly_cls' column
        timestamp (pd.Timestamp): Current timestamp for which to display risk assessment
    """
    def __init__(self, cls_predictions: pd.DataFrame, timestamp: pd.Timestamp, key: str = "predicitons_hourly_cls"):
        """
        Initialize the OverflowRiskometer component.

        Args:
            cls_predictions (pd.DataFrame): DataFrame containing CLS prediction values.
                                          Expected to have timestamps as index and
                                          'predicitons_hourly_cls' column with float values.
            timestamp (pd.Timestamp): The specific timestamp for which to display
                                    the risk assessment.
        """
        self.cls_predictions = cls_predictions
        self.timestamp = timestamp
        self.key = key

    def render(self):
        """
        Render the overflow riskometer component in Streamlit.

        This method creates a complete risk assessment display including:
        - Risk level categorization (LOW/MEDIUM/HIGH)
        - Color-coded visual indicators
        - Progress bar showing exact risk position
        - Explanatory tooltip
        - Current status and probability values

        The risk levels are determined as follows:
        - LOW RISK: 0-33% probability (Green)
        - MEDIUM RISK: 33-67% probability (Orange)
        - HIGH RISK: 67-100% probability (Red)

        Returns:
            None: Renders the component directly to Streamlit interface
        """
        # Add riskometer under the map
        st.markdown("---")

        # Get CLS prediction value for current timestamp to determine risk level
        cls_value = None
        if self.cls_predictions is not None and self.timestamp in self.cls_predictions.index:
            try:
                cls_value = self.cls_predictions.loc[self.timestamp, self.key]
                if pd.isna(cls_value):
                    cls_value = None
            except Exception:
                cls_value = None

        # Create riskometer using Streamlit components
        if cls_value is not None:
            try:
                # Normalize to 0-1 range
                normalized_value = max(0, min(1, float(cls_value)))

                # Determine risk zone and color
                if normalized_value < 0.33:
                    risk_zone = "LOW RISK"
                    zone_color = "#4CAF50"  # Green
                    risk_level = "🟢"
                elif normalized_value < 0.67:
                    risk_zone = "MEDIUM RISK"
                    zone_color = "#FF9800"  # Orange
                    risk_level = "🟡"
                else:
                    risk_zone = "HIGH RISK"
                    zone_color = "#F44336"  # Red
                    risk_level = "🔴"

                # Create riskometer using Streamlit components
                st.markdown("**Overflow Risk in the coming 2 hours at the Sewage Treatment Facility Location**")

                # Add tooltip with explanation
                with st.expander("ℹ️ What does this mean?", expanded=False):
                    st.markdown("""
                    **Overflow Risk Assessment:**
                    - **LOW RISK (0-33%)**: Minimal chance of overflow in the next 2 hours
                    - **MEDIUM RISK (33-67%)**: Moderate chance of overflow, monitoring recommended
                    - **HIGH RISK (67-100%)**: High chance of overflow, immediate attention required

                    The risk is calculated based on current filling levels, rainfall characteristics, and historical patterns.
                    """)

                # Risk meter visualization
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    if normalized_value < 0.33:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: {zone_color}; color: white; border-radius: 10px; font-weight: bold;'>{risk_level}<br>LOW</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: #e0e0e0; color: #666; border-radius: 10px;'>🟢<br>LOW</div>", unsafe_allow_html=True)

                with col2:
                    if 0.33 <= normalized_value < 0.67:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: {zone_color}; color: white; border-radius: 10px; font-weight: bold;'>{risk_level}<br>MEDIUM</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: #e0e0e0; color: #666; border-radius: 10px;'>🟡<br>MEDIUM</div>", unsafe_allow_html=True)

                with col3:
                    if normalized_value >= 0.67:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: {zone_color}; color: white; border-radius: 10px; font-weight: bold;'>{risk_level}<br>HIGH</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; padding: 20px; background: #e0e0e0; color: #666; border-radius: 10px;'>🔴<br>HIGH</div>", unsafe_allow_html=True)

                # Progress bar showing exact position
                st.progress(normalized_value)

                # Current status
                st.markdown(f"**Current Status:** {risk_zone}")
                st.markdown(f"**Overflow Probability:** {cls_value:.2f}")

            except Exception:
                st.error("Error loading risk data")
        else:
            st.info("No Overflow Probability predictions data available")
