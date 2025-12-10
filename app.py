import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# LOAD STEMMER (AMAN STREAMLIT CLOUD)
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# FUNGSI LOAD FILE → BASE64
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# ==========================================
# PATH FILE GAMBAR (SESUAI STRUKTUR GITHUB KAMU)
# ==========================================
BG_IMAGE_FILENAME = "gamabr"        # FILE (bukan folder)
EXTRA_IMAGE_FILENAME = "images.jpg" # FILE

BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)

# ==========================================
# CSS BACKGROUND
# ==========================================
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10,25,47,0.4), rgba(10,25,47,0.6)),
            url("data:image/*;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
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

# ==========================================
# CSS UI
# ==========================================
ui_style = """
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #ccd6f6;
}

.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100,255,218,0.3);
}

.sentiment-badge {
    font-size: 26px;
    font-weight: 700;
    padding: 12px 35px;
    border-radius: 50px;
    color: white;
    margin-top: 15px;
    text-align: center;
}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)

# ==========================================
# PREPROCESSING
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

# ==========================================
# LOAD MO
