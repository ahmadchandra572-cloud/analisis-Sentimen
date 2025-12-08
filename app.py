import streamlit as st
import joblib
import re
import string
import base64
import pandas as pd
import requests # <-- DIBUTUHKAN untuk mengunduh leksikon
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Jika ini digunakan

# ==========================================
# 0️⃣ PRE-LOAD LEXICON (Kamus Kata)
# ==========================================
# Unduh kamus leksikon positif dan negatif dari GitHub
# Menggunakan @st.cache_data agar hanya diunduh sekali
@st.cache_data
def load_lexicon():
    try:
        positive_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
        negative_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
        
        # Baca langsung dari URL
        positive_lexicon = set(pd.read_csv(positive_url, sep="\t", header=None)[0])
        negative_lexicon = set(pd.read_csv(negative_url, sep="\t", header=None)[0])
        return positive_lexicon, negative_lexicon
    except Exception as e:
        st.error(f"Gagal memuat leksikon (pastikan internet stabil saat deploy): {e}")
        return set(), set()

POSITIVE_LEXICON, NEGATIVE_LEXICON = load_lexicon()

# Fungsi untuk menentukan sentimen dan menghitung skornya
def determine_sentiment_lexicon(text):
    if isinstance(text, str):
        # Menggunakan teks yang sudah dibersihkan dan di-stem (dari preprocessing)
        positive_count = sum(1 for word in text.split() if word in POSITIVE_LEXICON)
        negative_count = sum(1 for word in text.split() if word in NEGATIVE_LEXICON)
        
        sentiment_score = positive_count - negative_count
        
        if sentiment_score > 0:
            sentiment = "Positif"
        elif sentiment_score < 0:
            sentiment = "Negatif"
        else:
            sentiment = "Netral"
        return sentiment_score, sentiment
    return 0, "Netral"


# ==========================================
# BACKGROUND STYLE (Menggunakan CSS lama Anda)
# ==========================================
# [Kode BACKGROUND STYLE ANDA di sini]
# (Kode yang sama persis seperti yang Anda berikan, termasuk get_base64_of_bin_file dan st.markdown)

def get_base64_of_bin_file(file_path):
    """Mengubah file gambar menjadi string Base64."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Asumsi nama file adalah 'gamabr'
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME) 


if BG_IMAGE_B64:
    BG_STYLE = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(17, 24, 39, 0.85);
        padding: 2rem;
        border-radius: 16px;
    }}
    h1, h2, h3, label, p {{
        color: #e5e7eb !important;
    }}
    </style>
    """
else:
    BG_STYLE = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f2937, #111827);
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(17, 24, 39, 0.85);
        padding: 2rem;
        border-radius: 16px;
    }
    h1, h2, h3, label, p {
        color: #e5e7eb !important;
    }
    </style>
    """

st.markdown(BG_STYLE, unsafe_allow_html=True)

# ==========================================
# STEMMER & PREPROCESSING
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode('ascii')
    if STEMMER:
        text = STEMMER.stem(text)
    return text.strip()


# ==========================================
# LOAD MODEL DAN VECTORIZER
# ==========================================
@st.cache_resource
def load_resources():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        models = {
            "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
            "SVM": joblib.load("model_SVM_GamGwo.pkl")
        }
        return vectorizer, models
    except Exception as e:
        st.error(f"Gagal load file: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# UI & LOGIC
# ==========================================
st.title("Sentiment Analyzer")
st.subheader("Text Sentiment Analysis App")

if VECTORIZER is None or MODELS is None:
    st.stop()

model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
input_text = st.text_area("Enter text here", height=120)

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.warning("Text cannot be empty!")
    else:
        # 1. Preprocessing
        clean_text = text_preprocessing(input_text)
        
        # 2. Prediction ML (Machine Learning)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        ml_prediction = model.predict(X)[0]

        # 3. Prediction Lexicon
        lex_score, lex_prediction = determine_sentiment_lexicon(clean_text)

        # 4. Tampilkan Hasil
        st.info(f"Teks Bersih: {clean_text}")

        # Tampilkan perbandingan dalam bentuk tabel/kolom
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🤖 HASIL MACHINE LEARNING ({model_choice})**")
            
            if ml_prediction.lower() == "positif":
                st.success(f"**SENTIMEN: {ml_prediction.upper()} ✅**")
            elif ml_prediction.lower() == "negatif":
                st.error(f"**SENTIMEN: {ml_prediction.upper()} ❌**")
            else:
                st.warning(f"**SENTIMEN: {ml_prediction.upper()} ⚪**")

        with col2:
            st.markdown(f"**📚 HASIL LEXICON (InSet)**")
            
            if lex_prediction.lower() == "positif":
                st.success(f"**SKOR: {lex_score} | SENTIMEN: {lex_prediction.upper()} ✅**")
            elif lex_prediction.lower() == "negatif":
                st.error(f"**SKOR: {lex_score} | SENTIMEN: {lex_prediction.upper()} ❌**")
            else:
                st.info(f"**SKOR: {lex_score} | SENTIMEN: {lex_prediction.upper()} ⚪**")
