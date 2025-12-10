import streamlit as st
import joblib
import re
import string
import base64
import pandas as pd
import os

from preprocessing import full_preprocessing
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocessing import (
    load_kamus,
    remove_emoji,
    remove_symbols,
    remove_numbers,
    remove_username,
    remove_url,
    remove_html
)

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 2. LOAD KAMUS
# ==========================================
KAMUS = load_kamus()

# ==========================================
# 3. IMAGE LOADER
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

BG_IMAGE_B64 = get_base64_of_bin_file("gamabr.jpg")
EXTRA_IMAGE_B64 = get_base64_of_bin_file("images.jpg")

# ==========================================
# 4. BACKGROUND & UI CSS
# ==========================================
background_css = f"""
<style>
.stApp {{
    background-image: 
        linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)),
        url("data:image/jpeg;base64,{BG_IMAGE_B64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    background-color: rgba(17, 34, 64, 0.25);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem 2rem;
}}
.image-left-style {{
    border-radius: 12px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# ==========================================
# 5. STEMMER
# ==========================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==========================================
# 6. FULL PREPROCESSING PIPELINE
# ==========================================
def full_preprocessing(text):
    original = text

    teks = remove_emoji(text)
    teks = remove_symbols(teks)
    teks = remove_numbers(teks)
    teks = remove_username(teks)
    teks = remove_url(teks)
    teks = remove_html(teks)

    casefold = teks.lower()
    words = casefold.split()

    # Normalisasi
    norm_words = [KAMUS.get(w, w) for w in words]
    normalized = " ".join(norm_words)

    # Stemming
    stemmed = stemmer.stem(normalized)

    return {
        "original": original,
        "cleaning": teks,
        "case_folding": casefold,
        "normalisasi": normalized,
        "stemming": stemmed
    }

# ==========================================
# 7. LOAD TF-IDF & MODEL
# ==========================================
@st.cache_resource
def load_resources():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    models = {
        "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
        "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
        "SVM": joblib.load("model_SVM_GamGwo.pkl")
    }
    return vectorizer, models

VECTORIZER, MODELS = load_resources()

# ==========================================
# 8. UI TITLE
# ==========================================
st.markdown("<h1 style='text-align:center;'>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)

# ==========================================
# 9. LAYOUT
# ==========================================
col_img, col_input = st.columns([1, 2])

with col_img:
    if EXTRA_IMAGE_B64:
        st.markdown('<div class="image-left-style">', unsafe_allow_html=True)
        st.image(f"data:image/jpeg;base64,{EXTRA_IMAGE_B64}", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_input:
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("💬 Masukkan komentar:", height=120)
    analyze_button = st.button("🔍 ANALISIS SEKARANG")

# ==========================================
# 10. PROCESS & RESULT
# ==========================================
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks")
        st.stop()

    # Preprocessing
    hasil = full_preprocessing(input_text)
    X = VECTORIZER.transform([hasil["stemming"]])

    model = MODELS[model_choice]
    prediction = model.predict(X)[0]

    # Badge
    if prediction.lower() == "positif":
        badge_bg = "linear-gradient(90deg, #059669, #34d399)"
        icon = "😊"
        label = "POSITIF"
    elif prediction.lower() == "negatif":
        badge_bg = "linear-gradient(90deg, #dc2626, #f87171)"
        icon = "😡"
        label = "NEGATIF"
    else:
        badge_bg = "linear-gradient(90deg, #64748b, #94a3b8)"
        icon = "😐"
        label = "NETRAL"

    # ==========================================
    # 11. TABEL PREPROCESSING
    # ==========================================
    st.markdown("## 🔍 Hasil Preprocessing")
    df_view = pd.DataFrame([{
        "Teks Asli": hasil["original"],
        "Cleaning": hasil["cleaning"],
        "Case Folding": hasil["case_folding"],
        "Normalisasi": hasil["normalisasi"],
        "Stemming": hasil["stemming"]
    }])
    st.dataframe(df_view, use_container_width=True)

    # ==========================================
    # 12. FINAL RESULT
    # ==========================================
    st.markdown(f"""
    <div style="text-align:center;margin-top:30px;">
        <div style="
            background:{badge_bg};
            padding:15px 40px;
            border-radius:50px;
            font-size:28px;
            font-weight:bold;
            color:white;">
            {icon} {label}
        </div>
    </div>
    """, unsafe_allow_html=True)
