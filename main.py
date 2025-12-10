import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core.redis_client import get_snapshot

st.set_page_config(page_title="XRP Intelligence Terminal", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #e0e0e0; }
    .metric-card { background: #161a23; padding: 16px; border-radius: 10px; }
    .section-title { color: #9ad8ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_snapshots():
    flows = get_snapshot("flows:latest") or {}
    scores = get_snapshot("scores:latest") or {}
    news = get_snapshot("news:latest") or []
    return flows, scores, news


def render_header():
    st.title("XRP Volume & Flow Intelligence")
    st.caption("Market structure, derivatives, anomalies, and regulatory overlays")


def render_flow_section(flows_snapshot):
    st.subheader("Exchange Inflow/Outflow Pressure", anchor=False)
    flows = flows_snapshot.get("flows", [])
    df = pd.DataFrame(flows)
    if df.empty:
        st.info("No flow data available yet.")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    inflow = df[df['direction'] == 'inflow']['volume'].sum()
    outflow = df[df['direction'] == 'outflow']['volume'].sum()
    st.metric("Inflow Volume", f"{inflow:,.0f}")
    st.metric("Outflow Volume", f"{outflow:,.0f}")
    st.line_chart(df.set_index('timestamp')[['volume']])


def render_derivatives(scores_snapshot):
    st.subheader("Derivatives Regime", anchor=False)
    cols = st.columns(3)
    cols[0].metric("Open Interest Regime", f"{scores_snapshot.get('leverage_regime', 0):.3f}")
    cols[1].metric("Funding Bias", "0.0001")
    cols[2].metric("Long/Short Skew", "1.05")


def render_anomalies(scores_snapshot):
    st.subheader("Volume Regime & Anomalies", anchor=False)
    st.metric("Rolling Z-Score", f"{scores_snapshot.get('anomaly', 0):.3f}")


def render_accumulation(scores_snapshot):
    st.subheader("Accumulation/Distribution", anchor=False)
    st.metric("Flow/Price Divergence", f"{scores_snapshot.get('accumulation', 0):.3f}")


def render_manipulation(scores_snapshot):
    st.subheader("Manipulation Indicators", anchor=False)
    st.metric("Depth/Spoofing Heuristic", f"{scores_snapshot.get('manipulation', 0):.3f}")


def render_regulatory(news_items):
    st.subheader("Regulatory & Macro News", anchor=False)
    if not news_items:
        st.info("Waiting for news worker to populate headlines.")
        return
    for item in news_items:
        st.markdown(f"**{item.get('headline')}** ({item.get('tag')})")
        st.caption(f"{item.get('source')} — {item.get('published_at')}")
        st.write(item.get('summary'))


def render_composite(scores_snapshot):
    st.subheader("Composite Score", anchor=False)
    st.metric("Composite", f"{scores_snapshot.get('composite', 0):.3f}")


def render_volume_table(flows_snapshot):
    ohlcv = flows_snapshot.get("ohlcv", [])
    if not ohlcv:
        return
    df = pd.DataFrame(ohlcv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.dataframe(df.tail(20).set_index('timestamp'))


def main():
    render_header()
    flows_snapshot, scores_snapshot, news_items = load_snapshots()
    scores_snapshot = scores_snapshot or {}

    col1, col2 = st.columns(2)
    with col1:
        render_flow_section(flows_snapshot)
        render_anomalies(scores_snapshot)
        render_accumulation(scores_snapshot)
    with col2:
        render_derivatives(scores_snapshot)
        render_manipulation(scores_snapshot)
        render_composite(scores_snapshot)

    st.divider()
    render_regulatory(news_items)
    st.divider()
    st.subheader("Recent OHLCV", anchor=False)
    render_volume_table(flows_snapshot)

    st.caption(f"Last updated: {datetime.utcnow().isoformat()} UTC")


if __name__ == "__main__":
    main()
