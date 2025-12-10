import streamlit as st
import joblib
import base64
import pandas as pd

from preprocessing import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ================================
# Config
# ================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ================================
# Load Kamus
# ================================
KAMUS = load_kamus()

# ================================
# Image Loader
# ================================
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

BG = get_base64("gamabr.jpg")
EXTRA = get_base64("images.jpg")

# ================================
# CSS
# ================================
st.markdown(f"""
<style>
.stApp {{
    background-image: 
        linear-gradient(rgba(10,20,40,0.5), rgba(10,20,40,0.6)),
        url("data:image/jpg;base64,{BG}");
    background-size: cover;
}}

.block-container {{
    background: rgba(17,34,64,0.3);
    padding: 2rem;
    border-radius: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ================================
# Stemmer
# ================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ================================
# Preprocessing Pipeline
# ================================
def preprocess(text):
    t = remove_emoji(text)
    t = remove_symbols(t)
    t = remove_numbers(t)
    t = remove_username(t)
    t = remove_url(t)
    t = remove_html(t)

    lower = t.lower().split()
    norm = [KAMUS.get(w, w) for w in lower]
    norm_text = " ".join(norm)
    stem = stemmer.stem(norm_text)

    return {
        "original": text,
        "cleaning": t,
        "case_folding": " ".join(lower),
        "normalisasi": norm_text,
        "stemming": stem
    }

# ================================
# Load Model + TF-IDF
# ================================
@st.cache_resource
def load_model():
    vec = joblib.load("tfidf_vectorizer.pkl")
    models = {
        "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
        "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
        "SVM": joblib.load("model_SVM_GamGwo.pkl")
    }
    return vec, models

VECTORIZER, MODELS = load_model()

# ================================
# UI
# ================================
st.title("🤖 Aplikasi Analisis Sentimen AI")

model_option = st.selectbox("Pilih Model", list(MODELS.keys()))
text = st.text_area("Masukkan Komentar")

if st.button("Analisis"):
    if text.strip() == "":
        st.warning("Masukkan teks dulu")
        st.stop()

    hasil = preprocess(text)
    X = VECTORIZER.transform([hasil["stemming"]])

    model = MODELS[model_option]

    # ================================
    # Override rule
    # ================================
    positif = ["baik", "bagus", "mantap", "hebat", "keren", "top"]
    negatif = ["buruk", "jelek", "parah", "busuk", "gagal"]

    stem_text = hasil["stemming"].strip()

    if stem_text in positif:
        pred = "positif"
    elif stem_text in negatif:
        pred = "negatif"
    else:
        pred = model.predict(X)[0]

    # ================================
    # Tabel
    # ================================
    st.subheader("Hasil Preprocessing")
    st.dataframe(pd.DataFrame([hasil]))

    # ================================
    # Output
    # ================================
    st.subheader("Hasil Analisis")
    st.success(f"Hasil Sentimen: {pred.upper()}")
