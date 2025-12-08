import streamlit as st
import joblib
import re
import string
import base64
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Jika ini digunakan


# ==========================================
# 0️⃣ KONFIGURASI HALAMAN & ENCODER
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# PENTING: NAMA FILE GAMBAR
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)


# ==========================================
# 1️⃣ CSS STYLE INJECTION (Biru Dongker Glassmorphism)
# ==========================================
# --- WARNA TEMA: BIRU DONGKER (DARK NAVY) ---
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(10, 25, 47, 0.90), rgba(10, 25, 47, 0.95)), 
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
        background: radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

ui_style = """
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }

/* Container Utama Glassmorphism */
.block-container {
    background-color: rgba(17, 34, 64, 0.4); 
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.1); 
    box-shadow: 0 10px 30px -10px rgba(2, 12, 27, 0.7);
    max-width: 680px;
}

/* Judul */
h1 {
    font-weight: 700;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

/* Input Area */
.stTextArea textarea {
    background-color: rgba(10, 25, 47, 0.6); 
    color: #e6f1ff;
    border: 1px solid #233554;
}

/* Kartu Hasil (Tengah & Pudar) */
.result-container { display: flex; justify-content: center; margin-top: 30px; }
.result-card {
    background: rgba(17, 34, 64, 0.5); 
    backdrop-filter: blur(10px); 
    border-radius: 16px;
    padding: 25px;
    width: 100%;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.sentiment-badge { 
    font-size: 24px; font-weight: 700; padding: 12px 35px; border-radius: 50px; 
    display: inline-block; color: white; margin-bottom: 20px; 
}

.clean-text-box {
    background: rgba(2, 12, 27, 0.4);
    padding: 15px;
    border-radius: 8px;
    font-size: 13px;
    color: #a8b2d1;
    font-family: 'Courier New', monospace;
    border-left: 3px solid #64ffda;
}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)


# ==========================================
# 2️⃣ DEPENDENCIES & PREPROCESSING
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
# 3️⃣ TAMPILAN UTAMA (UI)
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.error("⚠️ Sistem gagal dimuat. Cek file .pkl di repo.")
    st.stop()

# Layout Input
with st.container():
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=100)
    analyze_button = st.button("🔍 ANALISIS SEKARANG")

# Logika Hasil
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # 1. Preprocessing
        clean_text = text_preprocessing(input_text)
        
        # 2. Vectorization (TF-IDF)
        X = VECTORIZER.transform([clean_text])
        tfidf_shape = X.shape # Mendapatkan shape TF-IDF (1, max_features)
        
        # 3. Prediksi
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Warna Badge (Tetap elegan tapi mencolok di background gelap)
        if prediction.lower() == "positif":
            badge_bg = "linear-gradient(90deg, #059669, #34d399)" # Hijau
            icon = "😊"
            label = "POSITIF"
        elif prediction.lower() == "negatif":
            badge_bg = "linear-gradient(90deg, #dc2626, #f87171)" # Merah
            icon = "😡"
            label = "NEGATIF"
        else:
            badge_bg = "linear-gradient(90deg, #64748b, #94a3b8)" # Abu-abu
            icon = "😐"
            label = "NETRAL"

        # Tampilkan Kartu Hasil (Centered)
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <div class="result-label">HASIL PREDIKSI</div>
                
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                
                <div style="text-align: left; margin-bottom: 5px; font-size: 11px; color: #8892b0; margin-left: 5px;">
                    Teks Bersih:
                </div>
                <div class="clean-text-box">
                    {clean_text}
                </div>
                
                <div style="text-align: left; margin-bottom: 5px; font-size: 11px; color: #8892b0; margin-left: 5px;">
                    Vector Shape (TF-IDF):
                </div>
                <div class="clean-text-box" style="border-left: 3px solid #ff9900;">
                    {tfidf_shape}
                </div>
                
                <div class="model-info">
                    Algoritma: <b>{model_choice}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
