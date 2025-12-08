import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# 0️⃣ KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 1️⃣ FUNGSI BACKGROUND & GAYA (CSS PREMIUM)
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# NAMA FILE GAMBAR (Sesuai request: 'gamabr' tanpa ekstensi)
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- STYLE CSS KHUSUS ---
# 1. Background Image (TETAP SAMA, TIDAK DIUBAH)
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(10, 25, 47, 0.85), rgba(10, 25, 47, 0.95)), 
                          url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
else:
    background_css = "<style>.stApp {background: #0f172a;}</style>"

# 2. UI Styling (Font Keren & Tabel Rapi Tengah)
ui_style = """
<style>
/* Import Google Font: Poppins */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

/* Reset Font Global */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Container Utama (Glassmorphism) */
.block-container {
    background-color: rgba(255, 255, 255, 0.03); /* Transparan gelap */
    backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    max-width: 680px; /* Lebar dibatasi agar rapi di tengah */
}

/* Judul Utama */
h1 {
    font-weight: 800;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* Sub-teks */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 300;
    margin-bottom: 30px;
}

/* Text Area */
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.6);
    color: white;
    border: 1px solid #334155;
    border-radius: 12px;
    font-size: 14px;
}
.stTextArea textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
}

/* Tombol Analisis */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    padding: 12px 0;
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: 0.3s;
    margin-top: 10px;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
}

/* --- KARTU HASIL (CARD) YANG DI REQUEST --- */
.result-container {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

.result-card {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 20px;
    padding: 30px;
    width: 100%;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    animation: fadeIn 0.8s ease-out;
}

.result-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #94a3b8;
    margin-bottom: 15px;
}

.sentiment-badge {
    display: inline-block;
    padding: 12px 35px;
    border-radius: 50px;
    font-size: 24px;
    font-weight: 700;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.clean-text-box {
    background: rgba(0, 0, 0, 0.2);
    padding: 15px;
    border-radius: 10px;
    font-size: 13px;
    color: #bae6fd;
    font-family: 'Courier New', monospace;
    margin-bottom: 15px;
    border-left: 3px solid #38bdf8;
}

.model-info {
    font-size: 11px;
    color: #64748b;
    margin-top: 10px;
}

/* Animasi */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

# Terapkan Semua CSS
st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)


# ==========================================
# 2️⃣ LOAD DEPENDENCIES
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
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR dengan Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.error("⚠️ Sistem gagal dimuat. Cek file .pkl di repo.")
    st.stop()

# Layout Input yang Rapi
with st.container():
    model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=120)
    analyze_button = st.button("🔍 ANALISIS SEKARANG")

# Logika Hasil dengan Tampilan Kartu Tengah
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # Proses
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Tentukan Gaya Kartu Berdasarkan Hasil
        if prediction.lower() == "positif":
            badge_bg = "linear-gradient(135deg, #10b981, #34d399)" # Hijau
            icon = "😊"
            label = "POSITIF"
        elif prediction.lower() == "negatif":
            badge_bg = "linear-gradient(135deg, #ef4444, #f87171)" # Merah
            icon = "😡"
            label = "NEGATIF"
        else:
            badge_bg = "linear-gradient(135deg, #64748b, #94a3b8)" # Abu-abu
            icon = "😐"
            label = "NETRAL"

        # Tampilkan KARTU HASIL (HTML)
        # Kartu ini didesain agar teks & elemennya rata tengah (text-align: center)
        # dan kontainernya sendiri berada di tengah (margin auto).
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <div class="result-label">HASIL PREDIKSI</div>
                
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                
                <div style="text-align: left; margin-bottom: 5px; font-size: 11px; color: #94a3b8; margin-left: 5px;">
                    Teks Bersih (Preprocessed):
                </div>
                <div class="clean-text-box">
                    {clean_text}
                </div>
                
                <div class="model-info">
                    Dianalisis menggunakan algoritma <b>{model_choice}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
