import os
import streamlit as st
import plotly.express as px
from openai import OpenAI
client = OpenAI(
    api_key="REMOVED proj-OV3UkdEbvT9pYWszG7VzM_3rlxgaylBzEH5HRFsnFfRlIv2Fu1-1w3PDemkpWcK7a9HHFlVcyaT3BlbkFJt222ehlVCxa21gVbHqX-oDbD8h2mxMHy23a4C6QGPHFti8TuBLn1XT_jtWelnuBacjj8yYeX8A"
    )


from preprocessing import load_data, add_derived_fields, basic_validation_flags
from metrics import kpi_summary, compute_quality_score, delay_by_carrier, kpis_per_carrier
from model import load_model, predict_delay

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


CARRIER_COLORS = {
    "CMA CGM": "#d8345f",
    "Hapag-Lloyd": "#f57f17",
    "Maersk": "#1976d2",
    "ONE": "#9c27b0",
    "MSC": "#43a047"
}


@st.cache_resource
def get_model():
    return load_model("artifacts/delay_model.pkl")

# Load + prepare
df = load_data("data/shipments.csv")
df = add_derived_fields(df)
df = basic_validation_flags(df)

# Filters
st.sidebar.header("Filters")


selected_carriers = st.multiselect(
    "Carrier",
    list(CARRIER_COLORS.keys()),
    default=["CMA CGM", "Maersk"]
)
chip_container = st.container()
with chip_container:
    for carrier in selected_carriers:
        st.markdown(
            f"""
            <span style="
                background:{CARRIER_COLORS[carrier]};
                padding:4px 10px;
                border-radius:12px;
                margin-right:6px;
                color:white;
                font-size:13px;
            ">{carrier}</span>
            """,
            unsafe_allow_html=True
        )


df_f = df[df["carrier"].isin(selected_carriers)]

# KPIs
kpis = kpi_summary(df_f)
quality_score = compute_quality_score(df_f)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: metric_card("Avg Transit (days)", kpis["avg_transit_time"])
with c2: metric_card("Avg Delay (days)", kpis["avg_delay_days"])
with c3: metric_card("On Time %", kpis["on_time_ratio"])
with c4: metric_card("Shipments", kpis["shipments"])
with c5: metric_card("Data Quality", quality_score)

# Charts
st.subheader("Performance")

fig = px.bar(
    delay_by_carrier(df_f),
    x="carrier",
    y="avg_delay",
    color="carrier",
    color_discrete_map=CARRIER_COLORS,
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
    color="carrier",
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
        "filters": {"carrier": carrier},
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
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
    ])

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)