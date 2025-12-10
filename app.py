import streamlit as st
import joblib
import re
import string
import base64

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 1. FUNGSI LOAD GAMBAR
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Gambar Background (WAJIB file, bukan folder)
BG_IMAGE_FILENAME = "gamabr/background.jpg"
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# Gambar Samping
EXTRA_IMAGE_FILENAME = "images.jpg"
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)

# ==========================================
# 2. CSS BACKGROUND
# ==========================================
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image: 
            linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)),
            repeating-linear-gradient(
                45deg,
                rgba(100, 255, 218, 0.02),
                rgba(100, 255, 218, 0.02) 2px,
                transparent 2px,
                transparent 40px
            ),
            url("data:image/jpeg;base64,{BG_IMAGE_B64}");
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
    .stApp {
        background-image:
            repeating-linear-gradient(
                45deg,
                rgba(100, 255, 218, 0.02),
                rgba(100, 255, 218, 0.02) 2px,
                transparent 2px,
                transparent 40px
            ),
            radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

# ==========================================
# 3. UI STYLE
# ==========================================
ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #ccd6f6;
}

/* Container Utama */
.block-container {
    background-color: rgba(17, 34, 64, 0.2);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.08);
    max-width: 900px;
}

h1 {
    font-weight: 700;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 1px;
}

.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}

.sentiment-badge {
    font-size: 28px;
    font-weight: 700;
    padding: 15px 40px;
    border-radius: 50px;
    display: inline-block;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.result-container {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

.result-card {
    background: rgba(17, 34, 64, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px 25px;
    width: 100%;
    max-width: 400px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)

# ==========================================
# 4. STEMMER
# ==========================================
try:
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# ==========================================
# 5. PREPROCESSING
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
# 6. LOAD MODEL
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
        st.error(f"Gagal memuat sistem: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# 7. UI UTAMA
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.error("⚠️ Sistem gagal dimuat.")
    st.stop()

with st.container():
    col_img, col_input = st.columns([1, 2])

    with col_img:
        if EXTRA_IMAGE_B64:
            st.markdown('<div class="image-left-style">', unsafe_allow_html=True)
            st.image(f"data:image/jpeg;base64,{EXTRA_IMAGE_B64}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Gambar tambahan tidak ditemukan.")

    with col_input:
        model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
        input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=100)
        analyze_button = st.button("🔍 ANALISIS SEKARANG")

# ==========================================
# 8. PREDIKSI
# ==========================================
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])

        model = MODELS[model_choice]
        prediction = model.predict(X)[0]

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

        st.markdown(
            f"""
            <div class="result-container">
                <div class="result-card">
                    <div class="sentiment-badge" style="background: {badge_bg};">
                        {icon} &nbsp; {label}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
