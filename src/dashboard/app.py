"""Live transaction intake and fraud operations dashboard."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from src.common.config import settings

st.set_page_config(page_title="Fraud Ops Dashboard", page_icon="🛡️",
                   layout="wide")

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "electronics", "travel", "clothing",
    "pharmacy", "utilities", "entertainment", "online",
]
CHANNELS = ["mobile_app", "web", "pos", "contactless", "online"]


@st.cache_data(ttl=3)
def load_data():
    try:
        engine = create_engine(settings.postgres_dsn, pool_pre_ping=True)
        with engine.connect() as conn:
            cases = pd.read_sql(text("""
                SELECT case_id, txn_id, decision, p_fraud, amount, user_id,
                       created_ts, status, analyst_verdict
                FROM cases ORDER BY created_ts DESC LIMIT 200
            """), conn)
            scored = pd.read_sql(text("""
                SELECT event_ts, txn_id, amount, p_fraud, decision, model_version
                FROM scored_transactions
                WHERE event_ts > now() - interval '24 hours'
                ORDER BY event_ts DESC LIMIT 5000
            """), conn)
            kpi = pd.read_sql(text("""
                SELECT
                  COUNT(*) AS total_scored,
                  SUM((decision = 'BLOCK')::int) AS blocked,
                  SUM((decision = 'REVIEW')::int) AS review,
                  SUM((decision = 'APPROVE')::int) AS approved,
                  ROUND(AVG(p_fraud)::numeric, 4) AS avg_p_fraud
                FROM scored_transactions
                WHERE event_ts > now() - interval '24 hours'
            """), conn)
            open_cases = pd.read_sql(text(
                "SELECT COUNT(*) FROM cases WHERE status = 'OPEN'"), conn)
        return {
            "connected": True,
            "cases": cases,
            "scored": scored,
            "kpi": kpi.iloc[0],
            "open_cases": int(open_cases.iloc[0, 0]),
        }
    except Exception as error:
        return {
            "connected": False,
            "error": f"{error.__class__.__name__}: {str(error)[:200]}",
        }


def submit_transaction(values):
    txn_id = f"LIVE-{uuid4().hex[:12].upper()}"
    payload = {
        "txn_id": txn_id,
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "user_id": values["user_id"].strip(),
        "card_id": values["card_id"].strip(),
        "merchant_id": values["merchant_id"].strip(),
        "merchant_cat": values["merchant_cat"],
        "amount": float(values["amount"]),
        "currency": values["currency"].strip().upper(),
        "channel": values["channel"],
        "device_id": values["device_id"].strip(),
        "ip": values["ip"].strip(),
        "lat": values["lat"],
        "lon": values["lon"],
    }
    url = f"{settings.inference_url.rstrip('/')}/v1/transactions"
    response = httpx.post(url, json=payload, timeout=12.0)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned {response.status_code}: {detail}")
    return response.json()


st.title("🛡️ Fraud Ops — Live Transaction Pipeline")
st.caption("New transactions are sent to the API → Kafka → stream scoring → PostgreSQL.")

if st.button("Refresh live results"):
    st.cache_data.clear()
    st.rerun()

st.subheader("Submit a transaction into the live pipeline")
st.caption(
    "Submitting sends an unlabeled event to Kafka. The stream processor "
    "calculates recent behavior features, asks the model for a score, and "
    "stores the decision. This form is a demo event source, not a payment provider."
)
with st.form("transaction_intake"):
    col1, col2, col3 = st.columns(3)
    user_id = col1.text_input("User ID", value="U-LIVE-001")
    card_id = col2.text_input("Card ID", value="C-LIVE-001")
    merchant_id = col3.text_input("Merchant ID", value="M-LIVE-001")

    col4, col5, col6 = st.columns(3)
    amount = col4.number_input("Amount", min_value=0.01, value=35.0, step=1.0)
    merchant_cat = col5.selectbox("Merchant category", MERCHANT_CATEGORIES)
    channel = col6.selectbox("Channel", CHANNELS)

    col7, col8, col9 = st.columns(3)
    device_id = col7.text_input("Device ID", value="D-LIVE-001")
    currency = col8.text_input("Currency", value="USD", max_chars=3)
    ip = col9.text_input("IP (optional)", value="")

    col10, col11 = st.columns(2)
    lat = col10.number_input("Latitude (optional; 0 means omitted)", value=0.0)
    lon = col11.number_input("Longitude (optional; 0 means omitted)", value=0.0)
    submitted = st.form_submit_button("Send transaction for live scoring",
                                      type="primary")

if submitted:
    required = (user_id, card_id, merchant_id, device_id, currency)
    if not all(value.strip() for value in required):
        st.error("User, card, merchant, device, and currency are required.")
    else:
        try:
            accepted = submit_transaction({
                "user_id": user_id,
                "card_id": card_id,
                "merchant_id": merchant_id,
                "merchant_cat": merchant_cat,
                "amount": amount,
                "channel": channel,
                "device_id": device_id,
                "currency": currency,
                "ip": ip,
                "lat": lat or None,
                "lon": lon or None,
            })
            st.session_state["last_submission"] = accepted
            st.cache_data.clear()
            st.success(
                f"Queued {accepted['txn_id']} on {accepted['topic']} "
                f"(partition {accepted['partition']}, offset {accepted['offset']}). "
                "Refresh live results shortly to see its score."
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            st.error(f"Could not submit transaction: {error}")

last_submission = st.session_state.get("last_submission")
if last_submission:
    st.info(
        f"Last accepted event: {last_submission['txn_id']} "
        f"({last_submission['topic']} offset {last_submission['offset']}). "
        "Queue acceptance confirms Kafka received it; the final score appears "
        "after the stream processor handles it."
    )

def render_pipeline_results():
    data = load_data()
    if not data["connected"]:
        st.warning(
            "PostgreSQL is not connected, so live scored results and cases "
            "cannot be displayed. Transactions can be submitted only while "
            "the API and Kafka are reachable."
        )
        st.caption(f"Database status: {data['error']}")
        return

    st.success("PostgreSQL connected — results are from the live scoring pipeline.")
    k = data["kpi"]
    st.subheader("Pipeline activity — last 24 hours")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scored", int(k["total_scored"] or 0))
    c2.metric("Blocked", int(k["blocked"] or 0))
    c3.metric("Review", int(k["review"] or 0))
    c4.metric("Approved", int(k["approved"] or 0))
    c5.metric("Open cases", data["open_cases"])

    scored = data["scored"]
    left, right = st.columns(2)
    with left:
        st.caption("Live decision mix")
        if not scored.empty:
            st.bar_chart(scored["decision"].value_counts())
        else:
            st.info("No transactions have been scored in the last 24 hours.")
    with right:
        st.caption("Fraud score by transaction amount")
        if not scored.empty:
            st.scatter_chart(scored, x="amount", y="p_fraud", color="decision")

    st.subheader("Latest scored transactions")
    st.dataframe(scored, use_container_width=True, hide_index=True, height=320)
    st.subheader(f"Recent cases ({data['open_cases']} open)")
    if not data["cases"].empty:
        st.dataframe(data["cases"], use_container_width=True, hide_index=True,
                     height=320)
    else:
        st.info("No cases have been opened.")


if hasattr(st, "fragment"):
    st.fragment(run_every="3s")(render_pipeline_results)()
else:
    render_pipeline_results()
