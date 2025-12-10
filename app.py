import streamlit as st
import joblib
import re
import string
import base64
import os
import pandas as pd
from sklearn.metrics import classification_report

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
# FUNGSI BACA FILE → BASE64
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# ==========================================
# PATH FILE (SESUAI REPO KAMU)
# ==========================================
BG_IMAGE_FILENAME = "gamabr"
EXTRA_IMAGE_FILENAME = "images.jpg"

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
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# UI UTAMA
# ==========================================
st.markdown("<h1 style='text-align:center;'>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)

# STATUS FILE MODEL
st.subheader("📁 Status File Model")
for f in [
    "tfidf_vectorizer.pkl",
    "model_RF_GamGwo.pkl",
    "model_LR_GamGwo.pkl",
    "model_SVM_GamGwo.pkl"
]:
    if os.path.exists(f):
        st.success(f"{f} ditemukan ✅")
    else:
        st.error(f"{f} tidak ditemukan ❌")

if VECTORIZER is None or MODELS is None:
    st.stop()

col_img, col_input = st.columns([1, 2])

with col_img:
    if EXTRA_IMAGE_B64:
        st.image(f"data:image/jpeg;base64,{EXTRA_IMAGE_B64}", use_column_width=True)

with col_input:
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("Masukkan komentar", height=100)
    analyze_button = st.button("🔍 Analisis")

# ==========================================
# PREDIKSI
# ==========================================
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu")
    else:
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0].lower()

        st.subheader("✅ Hasil Prediksi")

        if prediction == "positif":
            st.markdown("<div class='sentiment-badge' style='background:#22c55e;'>✅ POSITIF</div>", unsafe_allow_html=True)
        elif prediction == "negatif":
            st.markdown("<div class='sentiment-badge' style='background:#ef4444;'>❌ NEGATIF</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sentiment-badge' style='background:#64748b;'>⚪ NETRAL</div>", unsafe_allow_html=True)

        # ================================
        # SIMPAN RIWAYAT PREDIKSI
        # ================================
        if "result_data" not in st.session_state:
            st.session_state.result_data = []

        st.session_state.result_data.append({
            "Komentar": input_text,
            "Hasil": prediction
        })

# ==========================================
# TABEL HASIL PREDIKSI
# ==========================================
st.subheader("📊 Riwayat Prediksi")

if "result_data" in st.session_state and len(st.session_state.result_data) > 0:
    df = pd.DataFrame(st.session_state.result_data)
    st.dataframe(df)
else:
    st.info("Belum ada data prediksi.")

# ==========================================
# TABEL EVALUASI MODEL
# ==========================================
st.subheader("📈 Evaluasi Model")

if st.button("Tampilkan Evaluasi"):
    # Contoh dummy evaluasi (bisa diganti dengan dataset asli)
    y_true = ["positif", "negatif", "netral", "positif", "negatif"]
    y_pred = ["positif", "negatif", "netral", "positif", "netral"]

    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)
