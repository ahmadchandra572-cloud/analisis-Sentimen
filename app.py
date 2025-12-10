import streamlit as st
import joblib
import re
import string
import base64
import os

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
# FUNGSI LOAD GAMBAR
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# ==========================================
# PATH BACKGROUND & IMAGE
# ==========================================
BG_IMAGE_FILENAME = "gamabr/background.jpg"   # ✅ background folder
LEFT_IMAGE_FILENAME = "images.jpg"            # ✅ gambar kiri

BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
LEFT_IMAGE_B64 = get_base64_of_bin_file(LEFT_IMAGE_FILENAME)

# ==========================================
# DEBUG STREAMLIT
# ==========================================
st.write("📂 Path background:", BG_IMAGE_FILENAME)
st.write("✅ File background ada?:", os.path.exists(BG_IMAGE_FILENAME))

# ==========================================
# CSS BACKGROUND (PALING STABIL)
# ==========================================
if BG_IMAGE_B64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(10,25,47,0.50), rgba(10,25,47,0.70)),
                url("data:image/jpeg;base64,{BG_IMAGE_B64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("❌ Background gagal dimuat, cek path file.")

# ==========================================
# UI STYLE
# ==========================================
ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }
.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}
.sentiment-badge {
    font-size: 28px; font-weight: 700; padding: 15px 40px;
    border-radius: 50px; color: white; margin: 10px 0;
}
</style>
"""
st.markdown(ui_style, unsafe_allow_html=True)

# ==========================================
# PREPROCESSING
# ==========================================
@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str): 
        return ""
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
# LOAD MODEL
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
# UI LAYOUT
# ==========================================
st.markdown("<h1 style='text-align:center;'>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.stop()

col_img, col_input = st.columns([1, 2])

with col_img:
    if LEFT_IMAGE_B64:
        st.image(f"data:image/jpeg;base64,{LEFT_IMAGE_B64}", use_column_width=True)

with col_input:
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("Masukkan komentar", height=100)
    analyze_button = st.button("🔍 Analisis")

# ==========================================
# PREDIKSI
# ==========================================
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0].lower()

        if prediction == "positif":
            st.success("✅ POSITIF")
        elif prediction == "negatif":
            st.error("❌ NEGATIF")
        else:
            st.info("⚪ NETRAL")
