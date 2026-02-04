"""
ETD Deviation Analysis Module
Handles ETD deviation predictions, visualizations, and shipment details table.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from etd_deviation_model import load_deviation_model, predict_etd_deviation


@st.cache_resource
def get_deviation_model():
    """Load the ETD deviation prediction model."""
    return load_deviation_model()


def add_deviation_predictions(df):
    """
    Add ETD deviation predictions to the dataframe.
    
    Args:
        df: DataFrame with shipment data
        
    Returns:
        DataFrame with added columns:
        - predicted_etd_deviation
        - actual_etd_deviation
    """
    deviation_model = get_deviation_model()
    df["predicted_etd_deviation"] = predict_etd_deviation(df, deviation_model)
    
    # Calculate actual ETD deviation if both ETD and ATD columns exist
    if "ATD" in df.columns and "ETD" in df.columns:
        df["actual_etd_deviation"] = (pd.to_datetime(df["ATD"]) - pd.to_datetime(df["ETD"])).dt.days
    elif "atd" in df.columns and "etd" in df.columns:
        df["actual_etd_deviation"] = (pd.to_datetime(df["atd"]) - pd.to_datetime(df["etd"])).dt.days
    else:
        # If columns don't exist, set to NaN
        df["actual_etd_deviation"] = pd.NA
    
    return df


def show_model_performance(df):
    """Display ETD Deviation Model performance metrics."""
    # Filter out outliers for more accurate metrics (same threshold as training)
    df_filtered = df[df["actual_etd_deviation"].abs() <= 100].copy()
    outliers_removed = len(df) - len(df_filtered)
    
    # Calculate various precision metrics on filtered data
    etd_mae = abs(df_filtered["predicted_etd_deviation"] - df_filtered["actual_etd_deviation"]).mean()
    etd_rmse = ((df_filtered["predicted_etd_deviation"] - df_filtered["actual_etd_deviation"]) ** 2).mean() ** 0.5
    
    # Calculate accuracy within tolerance windows
    within_1_day = (abs(df_filtered["predicted_etd_deviation"] - df_filtered["actual_etd_deviation"]) <= 1).mean() * 100
    within_3_days = (abs(df_filtered["predicted_etd_deviation"] - df_filtered["actual_etd_deviation"]) <= 3).mean() * 100
    within_7_days = (abs(df_filtered["predicted_etd_deviation"] - df_filtered["actual_etd_deviation"]) <= 7).mean() * 100
    
    st.subheader("ETD Deviation Model Performance")
    
    # Show outlier info if any were removed
    if outliers_removed > 0:
        st.info(f"📊 Metrics calculated on {len(df_filtered)} shipments ({outliers_removed} outliers with |deviation| > 100 days excluded)")
    
    # Show metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error", f"{etd_mae:.2f} days")
    with col2:
        st.metric("RMSE", f"{etd_rmse:.2f} days")
    with col3:
        st.metric("Predictions within ±3 days", f"{within_3_days:.1f}%")
    
    # Additional accuracy metrics
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Within ±1 day", f"{within_1_day:.1f}%")
    with col5:
        st.metric("Within ±7 days", f"{within_7_days:.1f}%")
    with col6:
        avg_deviation = df_filtered["actual_etd_deviation"].mean()
        st.metric("Avg Actual Deviation", f"{avg_deviation:.2f} days")
    
    st.markdown(f"""
    <div style="padding:12px; border-radius:8px; background:#e3f2fd; margin-top:10px;">
        <p style="margin:0; font-size:13px; color:#666;">Model uses: doc completeness (0-7), HS code completeness (0-2), port congestion (0-20), and carrier confirmation (0-1)</p>
    </div>
    """, unsafe_allow_html=True)


def plot_etd_deviation_scatter(df, color_sequence):
    """
    Create scatter plot comparing actual vs predicted ETD deviation.
    Filters out outliers (|actual_deviation| > 100 days) for better visualization.
    
    Args:
        df: DataFrame with predictions
        color_sequence: List of colors for carriers
        
    Returns:
        Plotly figure
    """
    # Filter out outliers for better visualization (same threshold as training)
    df_filtered = df[df["actual_etd_deviation"].abs() <= 100].copy()
    
    # Count removed outliers
    outliers_removed = len(df) - len(df_filtered)
    
    title = f"Actual vs Predicted ETD Deviation ({len(df_filtered)} shipments"
    if outliers_removed > 0:
        title += f", {outliers_removed} outliers removed)"
    else:
        title += ")"
    
    fig_etd_dev = px.scatter(
        df_filtered,
        x="actual_etd_deviation",
        y="predicted_etd_deviation",
        color="MasterCarrier",
        color_discrete_sequence=color_sequence,
        title=title,
        labels={
            "actual_etd_deviation": "Actual ETD Deviation (days)",
            "predicted_etd_deviation": "Predicted ETD Deviation (days)"
        }
    )
    # Add perfect prediction line
    if len(df_filtered) > 0:
        fig_etd_dev.add_shape(
            type="line",
            x0=df_filtered["actual_etd_deviation"].min(),
            y0=df_filtered["actual_etd_deviation"].min(),
            x1=df_filtered["actual_etd_deviation"].max(),
            y1=df_filtered["actual_etd_deviation"].max(),
            line=dict(color="red", dash="dash"),
        )
    return fig_etd_dev


def plot_deviation_distributions(df):
    """
    Create histogram plots showing distribution of actual vs predicted ETD deviation.
    
    Args:
        df: DataFrame with predictions
    """
    st.subheader("ETD Deviation Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist_actual = px.histogram(
            df,
            x="actual_etd_deviation",
            nbins=30,
            title="Actual ETD Deviation Distribution",
            labels={"actual_etd_deviation": "ETD Deviation (days)"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig_hist_actual.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_hist_actual, use_container_width=True)
    
    with col2:
        fig_hist_pred = px.histogram(
            df,
            x="predicted_etd_deviation",
            nbins=30,
            title="Predicted ETD Deviation Distribution",
            labels={"predicted_etd_deviation": "Predicted ETD Deviation (days)"},
            color_discrete_sequence=["#ff7f0e"]
        )
        fig_hist_pred.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_hist_pred, use_container_width=True)


def show_shipment_details_table(df):
    """
    Display shipment details table ordered by predicted ETD deviation.
    
    Args:
        df: DataFrame with shipment data and predictions
    """
    st.subheader("Shipment Details - Ordered by Predicted ETD Deviation")
    
    # Define columns we want to include - prioritize key info and predictions
    desired_columns = [
        "ID", "POL", "POD", "MasterCarrier",
        "ETD", "ATD",
        "predicted_etd_deviation", "actual_etd_deviation",
        "doc_completeness", "commodity_incomplete", "port_congestion", "carrier_confirmation_conf"
    ]
    
    # Debug: Check if predicted_etd_deviation exists
    if "predicted_etd_deviation" not in df.columns:
        st.warning("⚠️ predicted_etd_deviation column not found in data!")
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in desired_columns if col in df.columns]
    
    # Prepare table data
    table_df = df[available_columns].copy()
    
    # Create column name mapping
    column_mapping = {
        "ID": "Shipment ID",
        "POL": "Origin",
        "POD": "Destination",
        "MasterCarrier": "Carrier",
        "ETD": "Estimated Departure",
        "ATD": "Actual Departure",
        "predicted_etd_deviation": "Predicted Dev (days)",
        "actual_etd_deviation": "Actual Dev (days)",
        "doc_completeness": "Doc Complete",
        "commodity_incomplete": "HS Code Status",
        "port_congestion": "Port Cong.",
        "carrier_confirmation_conf": "Carrier Conf."
    }
    
    # Rename columns
    table_df.columns = [column_mapping.get(col, col) for col in available_columns]
    
    # Sort by predicted ETD deviation if the column exists
    if "Predicted Dev (days)" in table_df.columns:
        table_df = table_df.sort_values("Predicted Dev (days)", ascending=False)
    
    # Format numeric columns
    if "Predicted Dev (days)" in table_df.columns:
        table_df["Predicted Dev (days)"] = table_df["Predicted Dev (days)"].round(2)
    if "Actual Dev (days)" in table_df.columns:
        table_df["Actual Dev (days)"] = pd.to_numeric(table_df["Actual Dev (days)"], errors='coerce').fillna(0).astype(int)
    if "Carrier Conf." in table_df.columns:
        table_df["Carrier Conf."] = table_df["Carrier Conf."].round(2)
    
    # Format binary HS Code Status column (1=Incomplete, 0=Complete)
    if "HS Code Status" in table_df.columns:
        table_df["HS Code Status"] = table_df["HS Code Status"].map({1: 'Incomplete', 0: 'Complete', 1.0: 'Incomplete', 0.0: 'Complete'}).fillna('Incomplete')
    
    # Format date columns to show only date (not time)
    if "Estimated Departure" in table_df.columns:
        table_df["Estimated Departure"] = pd.to_datetime(table_df["Estimated Departure"]).dt.strftime('%Y-%m-%d')
    if "Actual Departure" in table_df.columns:
        table_df["Actual Departure"] = pd.to_datetime(table_df["Actual Departure"]).dt.strftime('%Y-%m-%d')
    
    # Display with styling
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )


def plot_data_quality_heatmap(df):
    """
    Create heatmap showing relationship between days before departure and data quality issues.
    X-axis: Days before/after ETD (deviation buckets)
    Y-axis: Data quality parameters
    Cell values: Number of shipments affected
    
    Args:
        df: DataFrame with shipment data and actual_etd_deviation
    """
    st.subheader("Data Quality Issues by ETD Deviation")
    
    # Create deviation buckets (days before/after estimated departure)
    df_heatmap = df.copy()
    
    # Define buckets: negative = early, positive = late
    bins = [-np.inf, -7, -4, -2, -1, 0, 1, 2, 4, 7, np.inf]
    labels = ['≤-7 days', '-7 to -4', '-4 to -2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '2 to 4', '4 to 7', '≥7 days']
    
    df_heatmap['deviation_bucket'] = pd.cut(df_heatmap['actual_etd_deviation'], bins=bins, labels=labels)
    
    # Parameters to analyze
    parameters = {
        'Document Completeness': 'doc_completeness',
        'Commodity Incomplete': 'commodity_incomplete',
        'Port Congestion': 'port_congestion',
        'Carrier Confirmation': 'carrier_confirmation_conf'
    }
    
    # Create matrix for heatmap
    heatmap_data = []
    
    for param_name, param_col in parameters.items():
        if param_col in df_heatmap.columns:
            # Count shipments in each bucket with issues
            if param_col == 'commodity_incomplete':
                # Binary: 1 = incomplete (issue), 0 = complete (no issue)
                counts = df_heatmap.groupby('deviation_bucket')[param_col].apply(lambda x: (x == 1).sum())
            elif param_col == 'carrier_confirmation_conf':
                # Lower values = less confident = issue
                counts = df_heatmap.groupby('deviation_bucket')[param_col].apply(lambda x: (x < 0.5).sum())
            else:
                # Higher values = more issues (doc_completeness, port_congestion)
                counts = df_heatmap.groupby('deviation_bucket')[param_col].apply(lambda x: (x > 0).sum())
            heatmap_data.append(counts.values)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=labels,
        y=list(parameters.keys()),
        colorscale='YlOrRd',
        text=heatmap_data,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Shipments<br>Affected")
    ))
    
    fig.update_layout(
        title="Data Quality Issues vs. ETD Deviation (Days Before/After Estimated Departure)",
        xaxis_title="Actual Deviation from ETD (days)",
        yaxis_title="Data Quality Parameter",
        height=400,
        xaxis={'side': 'bottom'},
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("📊 Heatmap shows count of shipments with data quality issues for each deviation bucket. "
              "Negative values = early departure, Positive = late departure")


def render_etd_deviation_analysis(df, color_sequence):
    """
    Main function to render all ETD deviation analysis components.
    
    Args:
        df: DataFrame with shipment data (must have predictions already added)
        color_sequence: List of colors for carriers in plots
    """
    # Show model performance
    show_model_performance(df)
    
    # Create two-column layout for scatter plots
    col1, col2 = st.columns(2)
    
    with col2:
        fig_etd_dev = plot_etd_deviation_scatter(df, color_sequence)
        st.plotly_chart(fig_etd_dev, use_container_width=True)
    
    # Show distribution plots
    plot_deviation_distributions(df)
    
    # Show data quality heatmap
    plot_data_quality_heatmap(df)
    
    # Show detailed shipment table
    show_shipment_details_table(df)
