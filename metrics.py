import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# -----------------------------
# Column schema adapters
# -----------------------------
def _schema_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aliases so downstream metrics can use a consistent schema.
    We prefer post-preprocessing column names, but gracefully alias if needed.
    """
    df = df.copy()

    # Preferred canonical names used in KPIs
    preferred = {
        "carrier": "Carrier",                       # raw input often upper-case C
        "master_carrier": "MasterCarrier",         # added by preprocessing
        "carrier_category": "CarrierCategory",     # added by preprocessing
        "origin_port": "POL",                      # UN/LOCODE 5-char for ocean
        "destination_port": "POD",                 # UN/LOCODE 5-char for ocean
        "etd": "etd",
        "atd": "atd",
        "eta": "eta",
        "ata": "ata",
        "estimated_delivery": "estimated_delivery",
        "actual_delivery": "actual_delivery",
        "transit_time_days": "transit_time_days",
        "delay_days": "delay_days",
        "flag_negative_transit": "flag_negative_transit",
        "flag_negative_delay": "flag_negative_delay",
    }

    # Build inverse map for aliasing if source column is missing but an alternative exists
    for canonical, source in preferred.items():
        if canonical not in df.columns:
            if source in df.columns:
                df[canonical] = df[source]
            else:
                # Fallbacks for common variants from legacy data
                # NOTE: do NOT overwrite if the canonical already existed
                if canonical == "carrier":
                    for alt in ["carrier_name", "Carrier Name", "CARRIER", "Carrier"]:
                        if alt in df.columns:
                            df[canonical] = df[alt]
                            break
                elif canonical in ("origin_port", "destination_port"):
                    # try common alternates
                    alt_opts = ["origin", "Origin", "ORG", "pol"] if canonical == "origin_port" else ["destination", "Destination", "DST", "pod"]
                    for alt in alt_opts:
                        if alt in df.columns:
                            df[canonical] = df[alt]
                            break
                elif canonical in ("etd", "atd", "eta", "ata"):
                    # sometimes mixed case
                    alt = canonical.upper()
                    if alt in df.columns:
                        df[canonical] = df[alt]
                elif canonical in ("estimated_delivery", "actual_delivery"):
                    # alias may already be set by preprocessing, otherwise leave missing
                    pass
                elif canonical in ("transit_time_days", "delay_days"):
                    # leave missing; downstream will handle safely
                    pass
                elif canonical in ("flag_negative_transit", "flag_negative_delay"):
                    # leave missing; downstream will handle safely
                    pass

    # Ensure boolean flags exist
    if "flag_negative_transit" not in df.columns:
        df["flag_negative_transit"] = False
    if "flag_negative_delay" not in df.columns:
        df["flag_negative_delay"] = False

    return df


# -----------------------------
# Utility helpers (null-safe)
# -----------------------------
def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    if series is None or series.empty:
        return default
    try:
        return float(series.dropna().mean()) if series.notna().any() else default
    except Exception:
        return default

def _safe_ratio(numer: pd.Series, denom_len: int) -> float:
    if denom_len <= 0:
        return 0.0
    return float(numer.sum()) / float(denom_len)


# -----------------------------
# Core quality score
# -----------------------------
def compute_quality_score(
    df: pd.DataFrame,
    *,
    # fields your downstream analytics and Ops need as a minimum viable record
    mandatory_fields: List[str] = None,
    # penalty weights to reflect business impact
    penalty_weights: Dict[str, float] = None
) -> float:
    """
    Compute a single quality score:
      score = present_ratio - weighted_penalties
    where present_ratio = average completeness across mandatory fields.
    Penalties reflect how harmful a record anomaly is for Ops.

    Returns a value in [0, 1], rounded to 2 decimals.
    """
    df = _schema_aliases(df)

    if mandatory_fields is None:
        # Business-rational minimum fields to trust a shipment record
        # - identity of carrier
        # - a lane (origin/destination)
        # - at least one set of operational timestamps (ATD/ATA) or their estimated counterparts
        mandatory_fields = [
            "carrier", "origin_port", "destination_port",
            "etd", "atd", "eta", "ata",
            "estimated_delivery", "actual_delivery"
        ]

    present_ratio = (
        df[mandatory_fields].notnull().mean(axis=0).mean()
        if set(mandatory_fields).issubset(df.columns)
        else 0.0
    )

    # Default penalty weights
    if penalty_weights is None:
        penalty_weights = {
            # negative transit is a hard data error (arrival before departure)
            "flag_negative_transit": 0.50,
            # very negative delay suggests wrong sign or wrong basis
            "flag_negative_delay": 0.25,
            # unmapped carrier undermines rollups and SLA reporting
            "unmapped_master": 0.25,
        }

    # Build penalties
    penalties = 0.0

    if "flag_negative_transit" in df.columns:
        penalties += penalty_weights.get("flag_negative_transit", 0.0) * float(df["flag_negative_transit"].mean())

    if "flag_negative_delay" in df.columns:
        penalties += penalty_weights.get("flag_negative_delay", 0.0) * float(df["flag_negative_delay"].mean())

    if "master_carrier" in df.columns:
        unmapped_ratio = float((df["master_carrier"].fillna("Unmapped") == "Unmapped").mean())
    elif "MasterCarrier" in df.columns:
        unmapped_ratio = float((df["MasterCarrier"].fillna("Unmapped") == "Unmapped").mean())
    else:
        unmapped_ratio = 1.0  # worst case, we cannot assess mapping

    penalties += penalty_weights.get("unmapped_master", 0.0) * unmapped_ratio

    score = max(0.0, min(1.0, present_ratio - penalties))
    return round(score, 2) * 100


# -----------------------------
# KPI summaries (business-facing)
# -----------------------------
def kpi_summary(
    df: pd.DataFrame,
    *,
    sla_days: int = 0,                 # SLA delay threshold (0 means on/before ETA/ETD)
    delay_tolerance_days: int = 1      # small tolerance window such as 1–2 days if needed
) -> Dict[str, float]:
    """
    Returns core KPIs null-safely. Uses transit_time_days and delay_days if present.
    """
    df = _schema_aliases(df)

    t_mean = _safe_mean(df.get("transit_time_days", pd.Series(dtype=float)))
    d_mean = _safe_mean(df.get("delay_days", pd.Series(dtype=float)))

    # Define on-time: delay_days <= delay_tolerance_days
    delay = df.get("delay_days", pd.Series(dtype=float))
    if delay is not None and not delay.empty and delay.notna().any():
        on_time = (delay <= delay_tolerance_days)
        on_time_ratio = round(float(on_time.mean()), 2) * 100
    else:
        on_time_ratio = 0.0

    # Mapping coverage for governance
    mc_col = "master_carrier" if "master_carrier" in df.columns else ("MasterCarrier" if "MasterCarrier" in df.columns else None)
    mapping_coverage = round(float((df[mc_col] != "Unmapped").mean()), 2) if mc_col else 0.0

    # Field completeness snapshot on most impactful fields
    completeness_fields = ["carrier", "origin_port", "destination_port", "etd", "atd", "eta", "ata", "estimated_delivery", "actual_delivery"]
    completeness_by_field = {}
    for c in completeness_fields:
        if c in df.columns:
            completeness_by_field[c] = round(float(df[c].notnull().mean()), 2)

    return {
        "shipments": int(len(df)),
        "avg_transit_time_days": round(t_mean, 2),
        "avg_delay_days": round(d_mean, 2),
        "on_time_ratio": on_time_ratio,
        "mapping_coverage": mapping_coverage,
        "completeness_by_field": completeness_by_field,
    }


def delay_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use MasterCarrier for business attribution, fallback to raw carrier if needed.
    """
    df = _schema_aliases(df)

    group_key = "master_carrier" if "master_carrier" in df.columns else ("MasterCarrier" if "MasterCarrier" in df.columns else "carrier")

    out = (
        df.groupby(group_key, as_index=False)
          .agg(
              avg_delay=("delay_days", "mean"),
              on_time_ratio=("delay_days", lambda s: float((s <= 0).mean()) if s.notna().any() else 0.0),
              shipments=("delay_days", "count")
          )
          .sort_values(["avg_delay", "on_time_ratio"], ascending=[True, False])
    )
    # round for presentation
    out["avg_delay"] = out["avg_delay"].round(2)
    out["on_time_ratio"] = out["on_time_ratio"].round(2)
    return out


def kpis_per_carrier(df: pd.DataFrame, *, sla_days: int = 0, delay_tolerance_days: int = 1) -> Dict[str, Dict]:
    """
    KPI bundle per MasterCarrier (preferred) or fallback to raw carrier.
    """
    df = _schema_aliases(df)
    key = "master_carrier" if "master_carrier" in df.columns else ("MasterCarrier" if "MasterCarrier" in df.columns else "carrier")

    result: Dict[str, Dict] = {}
    for carrier, grp in df.groupby(key):
        result[str(carrier)] = kpi_summary(grp, sla_days=sla_days, delay_tolerance_days=delay_tolerance_days)
    return result


def delay_by_lane(df: pd.DataFrame) -> pd.DataFrame:
    """
    Useful for Operations: lane-level view.
    Lane = POL -> POD (origin_port -> destination_port)
    """
    df = _schema_aliases(df)
    if not {"origin_port", "destination_port"}.issubset(df.columns):
        return pd.DataFrame(columns=["origin_port", "destination_port", "avg_delay", "on_time_ratio", "shipments"])

    out = (
        df.groupby(["origin_port", "destination_port"], as_index=False)
          .agg(
              avg_delay=("delay_days", "mean"),
              on_time_ratio=("delay_days", lambda s: float((s <= 0).mean()) if s.notna().any() else 0.0),
              shipments=("delay_days", "count")
          )
          .sort_values(["avg_delay", "on_time_ratio"], ascending=[True, False])
    )
    out["avg_delay"] = out["avg_delay"].round(2)
    out["on_time_ratio"] = out["on_time_ratio"].round(2)
    return out


def data_completeness_report(df: pd.DataFrame, fields: List[str] = None) -> pd.DataFrame:
    """
    Tabular completeness report for selected fields. Defaults to the key operational fields.
    """
    df = _schema_aliases(df)
    if fields is None:
        fields = ["carrier", "master_carrier", "carrier_category", "origin_port", "destination_port", "etd", "atd", "eta", "ata", "estimated_delivery", "actual_delivery"]

    existing = [f for f in fields if f in df.columns]
    if not existing:
        return pd.DataFrame(columns=["field", "completeness"])

    comp = pd.DataFrame({
        "field": existing,
        "completeness": [round(float(df[f].notnull().mean()), 2) for f in existing]
    }).sort_values("completeness", ascending=True)
    return comp

def risk_by_carrier(df: pd.DataFrame, proba_col: str = "predicted_delay_risk") -> pd.DataFrame:
    """
    Computes per-carrier average risk (mean probability).
    Requires df[proba_col] to exist.
    """
    df = _schema_aliases(df)
    key = "master_carrier" if "master_carrier" in df.columns else ("MasterCarrier" if "MasterCarrier" in df.columns else "carrier")

    if proba_col not in df.columns:
        raise ValueError(f"Column '{proba_col}' not found. Ensure classifier predictions are added first.")

    out = (
        df.groupby(key, as_index=False)
          .agg(
              avg_risk=(proba_col, "mean"),
              shipments=("delay_days", "count")
          )
          .sort_values("avg_risk", ascending=False)
    )
    out["avg_risk"] = out["avg_risk"].round(3)
    return out

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
    
    df_heatmap['deviation_bucket'] = pd.cut(df_heatmap['delay_days'], bins=bins, labels=labels)
    
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