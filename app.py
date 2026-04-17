import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
# ─────────────────────────────────────────────
# LOAD AND TRAIN MODEL
# We retrain on the full dataset at startup
# since we don't have a saved model file
# ─────────────────────────────────────────────
 
@st.cache_resource
def load_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    df = df.drop(columns=["name"])
    X = df.drop(columns=["status"])
    y = df["status"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline, X.columns.tolist()
 
model, feature_names = load_model()
 
# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
 
st.title("Parkinson's Disease Detection")
st.write("Enter the biomedical voice measurements below to get a prediction.")
 
st.markdown("---")
 
# ─────────────────────────────────────────────
# INPUT FORM — 22 voice features
# ─────────────────────────────────────────────
 
st.subheader("Voice Measurements")
 
# Default values taken from dataset mean values for reference
default_values = {
    "MDVP:Fo(Hz)"    : 154.23,
    "MDVP:Fhi(Hz)"   : 197.10,
    "MDVP:Flo(Hz)"   : 116.32,
    "MDVP:Jitter(%)" : 0.00622,
    "MDVP:Jitter(Abs)": 0.00004,
    "MDVP:RAP"       : 0.00330,
    "MDVP:PPQ"       : 0.00346,
    "Jitter:DDP"     : 0.00991,
    "MDVP:Shimmer"   : 0.02971,
    "MDVP:Shimmer(dB)": 0.28228,
    "Shimmer:APQ3"   : 0.01566,
    "Shimmer:APQ5"   : 0.01791,
    "MDVP:APQ"       : 0.02447,
    "Shimmer:DDA"    : 0.04698,
    "NHR"            : 0.02488,
    "HNR"            : 21.886,
    "RPDE"           : 0.49865,
    "DFA"            : 0.71822,
    "spread1"        : -5.68423,
    "spread2"        : 0.22697,
    "D2"             : 2.38177,
    "PPE"            : 0.20634,
}
 
# Render inputs in two columns for cleaner layout
col1, col2 = st.columns(2)
user_input = {}
 
for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        with col1:
            user_input[feature] = st.number_input(
                label=feature,
                value=float(default_values.get(feature, 0.0)),
                format="%.5f"
            )
    else:
        with col2:
            user_input[feature] = st.number_input(
                label=feature,
                value=float(default_values.get(feature, 0.0)),
                format="%.5f"
            )
 
st.markdown("---")
 
# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
 
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
 
    st.subheader("Prediction Result")
 
    if prediction == 1:
        st.error(f"Parkinson's Disease Detected")
        st.write(f"Confidence: **{probability[1]*100:.2f}%**")
    else:
        st.success(f"Healthy — No Parkinson's Detected")
        st.write(f"Confidence: **{probability[0]*100:.2f}%**")
 
    st.markdown("---")
    st.caption("⚠️ This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis.")