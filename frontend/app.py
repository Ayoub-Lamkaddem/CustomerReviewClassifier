# app.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()
BACKEND_URL = os.getenv("BACK_END_URL")

st.title("Text Classification Demo")
st.write("Compare BERT LoRA and SGD models")

user_input = st.text_area("Enter a sentence to classify:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": user_input}
            )
            response.raise_for_status()
            result = response.json()
            st.subheader("Predictions:")
            st.write(f"**BERT LoRA:** {result['bert_prediction']}")
            st.write(f"**SGD + TFIDF:** {result['sgd_prediction']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de communication avec le backend: {e}")


# streamlit run app.py