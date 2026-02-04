import os
import re
import json
import streamlit as st
import plotly.express as px
from openai import OpenAI
from pathlib import Path

key_path = Path(__file__).resolve().parent / "openai_key.txt"
OPENAI_API_KEY = key_path.read_text(encoding="utf-8").strip()

client = OpenAI(
    api_key=OPENAI_API_KEY
    )


from preprocessing import load_data, add_derived_fields, basic_validation_flags
from metrics import kpi_summary, compute_quality_score, delay_by_carrier, kpis_per_carrier
from model import load_model, predict_delay
from ui_tags import build_color_map, inject_multiselect_tag_css

st.set_page_config(layout="wide")

# Visual config

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

@st.cache_resource
def get_model():
    return load_model("artifacts/delay_model.pkl")

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
quality_score = compute_quality_score(df_f)

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1: metric_card("Avg Transit (days)", kpis["avg_transit_time_days"])
with c2: metric_card("Avg Delay (days)", kpis["avg_delay_days"])
with c3: metric_card("On Time %", kpis["on_time_ratio"])
with c4: metric_card("Shipments", kpis["shipments"])
with c5: metric_card("Data Quality", quality_score)

# Charts
st.subheader("Performance")


dbc = delay_by_carrier(df_f)  # columns: master_carrier, avg_delay, on_time_ratio, shipments

discrete_seq = [CARRIER_COLORS[c] for c in all_carriers if c in CARRIER_COLORS]

fig = px.bar(
    dbc,
    x="master_carrier",
    y="avg_delay",
    color="master_carrier",
    color_discrete_sequence=discrete_seq,
    title="Average Delay by Carrier"
)

fig.update_layout(
    showlegend=False,
    height=350,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# Model predictions
model = get_model()
df_f["predicted_delay_days"] = predict_delay(df_f, model)

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
        "kpis_overall": kpis,                  # baseline
        "quality_overall": quality_score,      # baseline
        "kpis_by_carrier": kpis_per_carrier(df_f),  # detailed deviations
    }


    prompt = f"""
    You are explaining logistic freight performance metrics.
    Context:
    {context}

    Question:
    {user_input}

    Answer clearly and concisely.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
    ])

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)