import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 1. LOAD STEMMER (AMAN STREAMLIT CLOUD)
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# ==========================================
# 2. FUNGSI LOAD FILE → BASE64
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# ==========================================
# 3. NAMA FILE SESUAI REPO KAMU
# ==========================================
BG_IMAGE_FILENAME = "gamabr"      # ✅ FILE background (bukan folder)
LEFT_IMAGE_FILENAME = "images.jpg" # ✅ FILE gambar kiri

BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
LEFT_IMAGE_B64 = get_base64_of_bin_file(LEFT_IMAGE_FILENAME)

# ==========================================
# 4. CSS BACKGROUND
# ==========================================
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10, 25, 47, 0.4), rgba(10, 25, 47, 0.6)),
            url("data:image/*;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
else:
    background_css = """
    <style>
    .stApp { background-color: #0a192f; }
    </style>
    """

st.markdown(background_css, unsafe_allow_html=True)

# ==========================================
# 5. LOAD MODEL
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
        st.error(f"Gagal memuat model: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# 6. PREPROCESSING
# ==========================================
@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode("ascii", "ignore").decode("ascii")
    if STEMMER:
        text = STEMMER.stem(text)
    return text.strip()

# ======
