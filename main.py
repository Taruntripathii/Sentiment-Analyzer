"""
Amazon Alexa Sentiment Analysis - Streamlit Frontend
Author: Tarun | CSE-AI 2nd Year, GITS Udaipur

Run backend first: flask --app api.py run --port=5000
Then run: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

PREDICTION_ENDPOINT = "http://127.0.0.1:5000/predict"

# --- Page config ---
st.set_page_config(page_title="Sentiment Predictor", page_icon="🎙️", layout="centered")

st.title("🎙️ Amazon Alexa Review — Sentiment Predictor")
st.markdown("Built by **Tarun** | CSE-AI, 2nd Year — GITS, Udaipur")
st.markdown("---")

# --- Bulk upload ---
st.subheader("📂 Bulk Prediction (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV with a 'Sentence' column. Results will be downloadable.",
    type="csv",
)

# --- Single text input ---
st.subheader("✏️ Single Text Prediction")
user_input = st.text_input("", placeholder="e.g. I love how easy Alexa is to set up!")

st.markdown("")

if st.button("🔍 Predict"):
    if uploaded_file is not None:
        response = requests.post(PREDICTION_ENDPOINT, files={"file": uploaded_file})

        if response.status_code == 200:
            response_bytes = BytesIO(response.content)
            result_df = pd.read_csv(response_bytes)

            st.success(f"Done! {len(result_df)} rows predicted.")
            st.dataframe(result_df.head(10))

            response_bytes.seek(0)
            st.download_button(
                label="⬇️ Download Full Predictions CSV",
                data=response_bytes,
                file_name="Predictions.csv",
                key="result_download_button",
            )
        else:
            st.error("Something went wrong with the bulk prediction. Is the Flask API running?")

    elif user_input.strip():
        try:
            response = requests.post(
                PREDICTION_ENDPOINT,
                json={"text": user_input.strip()}
            )
            result = response.json()

            sentiment = result.get("prediction", "Unknown")
            if sentiment == "Positive":
                st.success(f"✅ Sentiment: **{sentiment}**")
            else:
                st.error(f"🚨 Sentiment: **{sentiment}**")
        except Exception as e:
            st.error(f"Could not reach the API. Make sure Flask is running.\n\nError: {e}")
    else:
        st.warning("Please upload a CSV or enter some text before predicting.")

st.markdown("---")
st.caption("Model: XGBoost + CountVectorizer + MinMaxScaler | Dataset: Amazon Alexa Reviews (TSV)")
