import os
import re
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from pathlib import Path
from model import predict_prob_over_thresholds_from_regression, load_calibrated_prob_artifacts, load_model, predict_delay
from preprocessing import load_data, add_derived_fields, basic_validation_flags
from metrics import kpi_summary, compute_quality_score, delay_by_carrier, kpis_per_carrier, _schema_aliases, plot_data_quality_heatmap
from ui_tags import build_color_map, inject_multiselect_tag_css

key_path = Path(__file__).resolve().parent / "openai_key.txt"
OPENAI_API_KEY = key_path.read_text(encoding="utf-8").strip()

client = OpenAI(
    api_key=OPENAI_API_KEY
    )

st.set_page_config(layout="wide")


#
import pandas as pd

# --- Load the supplemental flags/metrics and merge into main df on ID ---

def load_flags_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV and returns only the columns needed for merge.
    Handles missing columns gracefully.
    """
    # Columns we want from the CSV (as provided in your sample)
    needed_cols = {
        "ID",
        "DTT",
        "doc_completeness",
        "commodity_incomplete",
        "port_congestion",
    }

    # Read with dtype hints to avoid silent parse issues
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only intersecting columns
    present = [c for c in needed_cols if c in df.columns]
    if "ID" not in present:
        raise ValueError("The CSV does not include 'ID', which is required for merging.")

    # De-duplicate by ID, prefer the last occurrence (or adjust as needed)
    # If there may be multiple rows per ID with different values, decide an aggregation strategy.
    df_flags = (
        df[present]
        .drop_duplicates(subset=["ID"], keep="last")
        .reset_index(drop=True)
    )
    return df_flags


def merge_flags_on_id(main_df: pd.DataFrame, flags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge the five columns into the main dataframe on ID.
    """
    if "ID" not in main_df.columns:
        raise ValueError("main_df must contain 'ID' for merging.")
    # Use suffixes to avoid accidental collisions, then clean up if needed
    merged = main_df.merge(flags_df, on="ID", how="left")
    return merged

def build_llm_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slice the dataframe to include only end-user-relevant columns.
    Removes timestamps and engineering columns.
    Ensures the function ALWAYS returns a valid DataFrame.
    """

    # Normalize column names if you already have a helper:
    # df2 = _schema_aliases(df.copy())
    # If not needed, keep a simple copy:
    df2 = df.copy()

    # Columns that matter for conversational insights
    cols = [
        "ID",
        "master_carrier",
        "carrier_category",
        "origin_port",
        "destination_port",
        "delay_days",
        "predicted_delay_risk",
        # Newly merged columns:
        "DTT",
        "doc_completeness",
        "commodity_incomplete",
        "port_congestion",
        "carrier_confirmation_conf",
    ]

    # Only keep columns that exist in df2
    cols = [c for c in cols if c in df2.columns]

    # Guarantee output is a DataFrame (even if empty)
    if not cols:
        return pd.DataFrame()

    # Sort by risk if risk exists
    if "predicted_delay_risk" in df2.columns:
        df2 = df2.sort_values("predicted_delay_risk", ascending=False)

    # Return curated subset, capped to 100 rows for context efficiency
    return df2[cols].head(150)

def metric_card(title, value, color="#222", bg="#f8f9fa"):
    st.markdown(
        f"""
        <div style="
            padding:16px;
            border-radius:8px;
            background:{bg};
            width:150px;
            box-shadow:0 1px 4px rgba(0,0,0,0.12);
            text-align:center;
        ">
            <div style="font-size:12px;color:#666;">{title}</div>
            <div style="font-size:28px;font-weight:600;color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- color rules for risk ---
def risk_color(val: float) -> str:
    """
    Input is a probability in [0,1], not a percent.
    Bands:
      green  : val < 0.10
      yellow : 0.10 <= val < 0.31   # covers 10.00%..30.99%
      red    : val >= 0.31
    If you want 10.00% to be green, change the second line to: if val < 0.11:
    """
    if pd.isna(val):
        return ""
    if val < 0.10:
        return "background-color:#e6f4ea; color:#1f5133;"   # green bg, dark green text
    if val < 0.31:
        return "background-color:#fff4e5; color:#8a4b08;"   # yellow bg, brown text
    return "background-color:#fde7e9; color:#6e1b1f;"       # red bg, dark red text


def predict_risk(df):
    arts = get_prob_artifacts()
    T = 5.0
    probs = predict_prob_over_thresholds_from_regression(df, arts, thresholds_days=T)
    return probs[f"p_over_T{T:g}"]

def per_carrier_model_metrics(df):
    """
    df must contain:
      - MasterCarrier
      - predicted_delay_days
      - predicted_delay_risk
      - delay_days
    """
    # model MAE per carrier
    def mae(a, b):
        return (a - b).abs().mean()

    grouped = df.groupby("MasterCarrier")
    
    out = grouped.apply(
        lambda g: pd.Series({
            "shipments": len(g),
            "actual_avg_delay": g["delay_days"].mean(),
            "predicted_avg_delay": g["predicted_delay_days"].mean(),
            "predicted_avg_risk": g["predicted_delay_risk"].mean(),
            "mae_delay": mae(g["delay_days"], g["predicted_delay_days"]),
            "on_time_ratio": (g["delay_days"] <= 0).mean()
        })
    ).reset_index()

    return out

def on_time_by_carrier(df):
    return (
        df.groupby("MasterCarrier")
          .apply(lambda g: float((g["delay_days"] <= 0).mean()))
          .to_dict()
    )


# @st.cache_resource
# def get_classifier_bundle():
#     return load_classifiers_from_dir("artifacts/classifiers")


@st.cache_resource
def get_model():
    return load_model("artifacts/delay_model.pkl")

@st.cache_resource
def get_prob_artifacts():
    return load_calibrated_prob_artifacts(
        base_model_path="artifacts/delay_model.pkl",
        resid_model_path="artifacts/resid_model.pkl",
        calibrators_dir="artifacts/calibrators",
    )


# Load + prepare
df = load_data("data/shipments_with_master_carrier.csv")
df = add_derived_fields(df)
df = basic_validation_flags(df)


# Derive current carrier set from MasterCarrier
all_carriers = sorted(df["MasterCarrier"].dropna().astype(str).unique().tolist())

# Optional per-brand overrides
preferred_colors = {
    # "Maersk": "#1976d2",
}

# Build capped color map (≤10 colors, cycle if more)
CARRIER_COLORS = build_color_map(
    all_carriers,
    preferred=preferred_colors,
    max_colors=10,       # hard cap
    palette=None,        # use default PALETTE_10
    cycle=True           # cycle beyond 10
)

inject_multiselect_tag_css(st, CARRIER_COLORS)


# Filters
with st.sidebar:
    st.header("Filters")
    default_vals = [c for c in ["MSC", "Maersk"] if c in all_carriers] or None

    selected_carriers = st.multiselect(
        "Carrier",
        all_carriers,
        default=default_vals
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# Apply filter by MasterCarrier
df_f = df[df["MasterCarrier"].isin(selected_carriers)].copy() if selected_carriers else df.copy()

# KPIs
kpis = kpi_summary(df_f)
st.subheader("Performance Over All Selected Carriers")
quality_score = compute_quality_score(df_f)

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1: metric_card("Avg Transit (days)", kpis["avg_transit_time_days"])
with c2: metric_card("Avg Delay (days)", kpis["avg_delay_days"])
with c3: metric_card("On Time %", kpis["on_time_ratio"])
with c4: metric_card("Shipments", kpis["shipments"])
with c5: metric_card("Data Quality", quality_score)



dbc = delay_by_carrier(df_f)  # columns: master_carrier, avg_delay, on_time_ratio, shipments

discrete_seq = [CARRIER_COLORS[c] for c in all_carriers if c in CARRIER_COLORS]

# Model predictions
model = get_model()
df_f["predicted_delay_days"] = predict_delay(df_f, model)
df_f["predicted_delay_risk"] = predict_risk(df_f)
carrier_perf = per_carrier_model_metrics(df_f)

# sort by highest risk first
df_ship = df_f.sort_values("predicted_delay_risk", ascending=False).copy()

# normalize column names first
df_ship = _schema_aliases(df_f.copy())

# ------------------------------------------------------------------
# NEW: Merge DTT / doc / congestion / commodity / carrier_conf flags
# ------------------------------------------------------------------
try:
    flags_df = load_flags_csv("data/processed_shipments_with_synthetic_cols.csv")
    df_ship = merge_flags_on_id(df_ship, flags_df)
except Exception as e:
    st.warning(f"Flag‑file merge skipped: {e}")
# ------------------------------------------------------------------
# adding heatmap across all selected shipments
plot_data_quality_heatmap(df_ship)

st.subheader("Per‑Carrier Performance Snapshot")

for _, row in carrier_perf.iterrows():
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    carrier = row["MasterCarrier"]
    color = CARRIER_COLORS.get(carrier, "#444")

    with c1:
        metric_card(carrier, f"{int(row['shipments'])}", color=color, bg="#f2f2f2")

    with c2:
        metric_card("Risk >5d", f"{row['predicted_avg_risk']:.2%}", color=color)

    with c3:
        metric_card("On Time %", f"{row['on_time_ratio']:.1%}", color=color)
    
    with c4:
        metric_card("Avg Delay", f"{row['actual_avg_delay']:.2f}d", color=color)
    
    # spacing between carrier rows
    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

st.subheader("Shipment Risk Table")

# pick the columns you want in the shipment table
candidate_cols = [
    "ID",
    "master_carrier",
    "carrier",
    "origin_port", "destination_port",
    "etd", "atd", "eta", "ata",
    "delay_days",
    "predicted_delay_days",
    "predicted_delay_risk",
    "DTT",
    "doc_completeness",
    "commodity_incomplete",
    "port_congestion",
    "carrier_confirmation_conf",
]

# only keep columns that actually exist
display_cols = [c for c in candidate_cols if c in df_ship.columns]

# sort by risk
df_ship = df_ship.sort_values("predicted_delay_risk", ascending=False)

# format + color
styled = (
    df_ship[display_cols]
    .style
    .applymap(risk_color, subset=["predicted_delay_risk"])
    .format({
        "predicted_delay_risk": "{:.2%}",
        "predicted_delay_days": "{:.2f}",
        "delay_days": "{:.2f}",
    })
)

st.dataframe(styled, use_container_width=True)

fig_pred = px.scatter(
    df_f,
    x="delay_days",
    y="predicted_delay_days",
    color="MasterCarrier",
    color_discrete_sequence=discrete_seq,
    title="Actual vs Predicted Delay"
)
st.plotly_chart(fig_pred, use_container_width=True)

# Chat
st.subheader("Ask me about the data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Why does this carrier look worse?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    context = {
        "filters": {"MasterCarrier": selected_carriers},
        "kpis_overall": kpis,
        "quality_overall": quality_score,
        "on_time_by_carrier": on_time_by_carrier(df_f),
        "model_metrics_by_carrier": carrier_perf.to_dict(orient="index"),
        "shipments": build_llm_shipments(df_ship).to_dict(orient="records"),
    }

    prompt = f"""
    You are a professional AI assistant specialized in maritime freight logistics. Respond in a serious, clear, and concise manner, suitable for frequent use in a business environment. Avoid humor, metaphors, or informal language. Focus on providing accurate, actionable, and direct insights. Note the 'predicted_delay_risk' means probability of shipment being delayed by 5 days or more. If asked about ports of origins from the shipping table by the user, please provide references in the form of select rows in question if they are available. If the data is not available in the dataset, then please refer to the aggregate metrics.

    Context:
    {context}

    Question:
    {user_input}

    Answer in a professional and businesslike tone.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
    ])

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # Which carrier is fastest from hamburg to shanghai based on the current shipment table

    # Identify most common routes for each carrier from the table, give me counts for each route so I know for sure

    # Explain high delay risk for this shipment number: 