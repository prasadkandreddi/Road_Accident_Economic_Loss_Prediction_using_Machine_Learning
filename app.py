import streamlit as st
import pickle
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Legal Case Classification System",
    layout="centered"
)

# ===============================
# LOAD MODEL & TRANSFORMER
# ===============================
@st.cache_resource
def load_artifacts():
    with open("NaiveBayes_model_1.pkl", "rb") as f:
        model = pickle.load(f)

    with open("transformer_1.pkl", "rb") as f:
        transformer = pickle.load(f)

    return model, transformer


model, transformer = load_artifacts()

# ===============================
# TITLE
# ===============================
st.title("⚖️ Legal Case Classification System")
st.caption("Machine Learning | NLP | Naive Bayes")

st.markdown(
    """
    Enter a **court case description** below and the model will predict  
    the **type of legal case**.
    """
)

# ===============================
# INPUT
# ===============================
case_text = st.text_area(
    "📝 Case Description",
    height=180,
    placeholder="Enter court case details here..."
)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Case Type"):
    if case_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = transformer.transform([case_text])
        prediction = model.predict(X)[0]

        st.success(f"📌 Predicted Case Type: **{prediction}**")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Deployed on Hugging Face Spaces | Streamlit")
