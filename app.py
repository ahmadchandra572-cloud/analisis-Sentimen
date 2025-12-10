import streamlit as st
import joblib
import base64
import pandas as pd
from preprocessing import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ====================================
# CONFIG
# ====================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ====================================
# LOAD KAMUS
# ====================================
KAMUS = load_kamus()

# ====================================
# IMAGE LOADER
# ====================================
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

BG = get_base64("gamabr.jpg")
EXTRA = get_base64("images.jpg")

# ====================================
# CSS BACKGROUND
# ====================================
st.markdown(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(10,25,47,0.5), rgba(10,25,47,0.6)),
        url("data:image/jpeg;base64,{BG}");
    background-size: cover;
    background-attachment: fixed;
}}

.block-container {{
    background: rgba(17,34,64,0.35);
    backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 20px;
}}

.image-left-style {{
    border-radius: 12px;
    border: 3px solid #64ffda;
}}
</style>
""", unsafe_allow_html=True)

# ====================================
# STEMMER
# ====================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ====================================
# PREPROCESSING PIPELINE
# ====================================
def full_preprocessing(text):
    teks = remove_emoji(text)
    teks = remove_symbols(teks)
    teks = remove_numbers(teks)
    teks = remove_username(teks)
    teks = remove_url(teks)
    teks = remove_html(teks)

    low = teks.lower().split()
    norm = [KAMUS.get(w, w) for w in low]
    norm_text = " ".join(norm)
    stem = stemmer.stem(norm_text)

    return {
        "original": text,
        "cleaning": teks,
        "case_folding": " ".join(low),
        "normalisasi": norm_text,
        "stemming": stem
    }

# ====================================
# LOAD MODEL + TFIDF
# ====================================
@st.cache_resource
def load_all():
    vec = joblib.load("tfidf_vectorizer.pkl")
    models = {
        "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
        "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
        "SVM": joblib.load("model_SVM_GamGwo.pkl")
    }
    return vec, models

VECTORIZER, MODELS = load_all()

# ====================================
# UI TITLE
# ====================================
st.markdown("<h1 style='text-align:center;'>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)

col_img, col_input = st.columns([1,2])

with col_img:
    if EXTRA:
        st.markdown('<div class="image-left-style">', unsafe_allow_html=True)
        st.image(f"data:image/jpeg;base64,{EXTRA}", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_input:
    algo = st.selectbox("Pilih Algoritma", list(MODELS.keys()))
    text = st.text_area("Masukkan Komentar")
    btn = st.button("🔍 Analisis")

# ====================================
# PROSES
# ====================================
if btn:
    if text.strip() == "":
        st.warning("Masukkan teks!")
        st.stop()

    hasil = full_preprocessing(text)
    X = VECTORIZER.transform([hasil["stemming"]])

    model = MODELS[algo]

    # Manual override dengan kata tambahan
    positif = ["baik", "bagus", "mantap", "hebat", "cantik", "indah", "luar biasa", "menarik"]
    negatif = ["buruk", "jelek", "parah", "gagal", "jelek banget", "menghina", "menyedihkan"]

    stem = hasil["stemming"].strip()

    # Cek jika ada kata positif/negatif di teks
    if any(word in stem.split() for word in positif):
        pred = "positif"
    elif any(word in stem.split() for word in negatif):
        pred = "negatif"
    else:
        pred = model.predict(X)[0]

    # TABEL
    st.subheader("Hasil Preprocessing")
    st.dataframe(pd.DataFrame([hasil]))

    # BADGE
    if pred == "positif":
        bg = "green"
        icon = "😊"
    elif pred == "negatif":
        bg = "red"
        icon = "😡"
    else:
        bg = "gray"
        icon = "😐"

    st.markdown(f"""
    <div style="text-align:center;margin-top:20px;">
        <div style="background:{bg};padding:15px 40px;
        border-radius:50px;font-size:25px;color:white;">
        {icon} {pred.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)
