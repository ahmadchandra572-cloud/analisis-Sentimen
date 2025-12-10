import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# LOAD STEMMER
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    STEMMER = StemmerFactory().create_stemmer()
except:
    STEMMER = None

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sentimen DPR", page_icon="🤖", layout="centered")

# ==========================================
# LOAD FILE → BASE64
# ==========================================
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

BG = get_base64("gamabr")
IMG = get_base64("images.jpg")

# ==========================================
# CSS BACKGROUND
# ==========================================
if BG:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10,25,47,0.3), rgba(10,25,47,0.6)),
            url("data:image/*;base64,{BG}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.image-box {
    border-radius: 12px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100,255,218,0.35);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
vectorizer = joblib.load("tfidf_vectorizer.pkl")
models = {
    "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
    "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
    "SVM": joblib.load("model_SVM_GamGwo.pkl")
}

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    if STEMMER:
        text = STEMMER.stem(text)
    return text.strip()

# ==========================================
# UI ONLY (SESUAI YANG KAMU MAU)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    if IMG:
        st.image(f"data:image/jpeg;base64,{IMG}", use_column_width=True, output_format="auto")

with col2:
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(models.keys()))
    text = st.text_area("Masukkan komentar", height=120)
    btn = st.button("🔍 Analisis")

# ==========================================
# HASIL
# ==========================================
if btn and text.strip():
    clean = preprocess(text)
    X = vectorizer.transform([clean])
    pred = models[model_choice].predict(X)[0]

    if pred.lower() == "positif":
        st.success("✅ POSITIF")
    elif pred.lower() == "negatif":
        st.error("❌ NEGATIF")
    else:
        st.info("⚪ NETRAL")
