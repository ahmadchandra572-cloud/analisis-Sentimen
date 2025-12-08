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
    layout="centered" # Membuat layout lebih 'kecil' dan fokus di tengah
)

# ==========================================
# 1️⃣ FUNGSI UTILITAS (GAMBAR & PREPROCESSING)
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Asumsi nama file gambar Anda
BG_IMAGE_FILENAME = "gamabr.jpg" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- CSS STYLING (The Magic) ---
# Mengatur background biru, tabel tengah, dan header keren
if BG_IMAGE_B64:
    BG_CSS = f"""
    <style>
    /* 1. Background Image dengan Overlay Biru Gelap */
    .stApp {{
        background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.9)), 
                          url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* 2. Mengatur Container Utama agar Rapi */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        max-width: 700px; /* Memperkecil lebar aplikasi agar pas */
    }}

    /* 3. Header Styling */
    h1 {{
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica', sans-serif;
        font-weight: 800;
        text-align: center;
        padding-bottom: 20px;
    }}
    
    h3, p, label {{
        color: #e0e7ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 4. Text Area Styling */
    .stTextArea textarea {{
        background-color: rgba(0, 0, 0, 0.3);
        color: white;
        border-radius: 10px;
        border: 1px solid #4b5563;
    }}

    /* 5. Tabel/Kartu Hasil yang Sejajar & Tengah */
    .result-card {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        animation: fadeIn 1s;
    }}
    
    .sentiment-box {{
        font-size: 24px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 10px;
        color: white;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }}

    /* Animasi Halus */
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """
    st.markdown(BG_CSS, unsafe_allow_html=True)

# --- FUNGSI STEMMER ---
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# --- FUNGSI PREPROCESSING ---
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
# 2️⃣ LOAD MODEL
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
# 3️⃣ USER INTERFACE (LAYOUT)
# ==========================================

# -- HEADER --
st.markdown("<h1>📊 ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8;'>Deteksi Opini Publik tentang Gaji DPR menggunakan Optimasi GAM-GWO</p>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.stop()

# -- INPUT AREA --
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("🤖 Pilih Model", list(MODELS.keys()))
    with col2:
        st.write("") # Spacer

    input_text = st.text_area("✍️ Masukkan komentar di sini...", height=100)

    # Tombol di tengah
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        analyze_button = st.button("🚀 ANALISIS SEKARANG", use_container_width=True)

# -- LOGIKA & TAMPILAN HASIL --
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks terlebih dahulu!")
    else:
        # Proses
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Tentukan Warna & Ikon berdasarkan hasil
        if prediction.lower() == "positif":
            bg_color = "linear-gradient(90deg, #00b09b, #96c93d)" # Hijau Segar
            icon = "✅"
            label = "POSITIF"
        elif prediction.lower() == "negatif":
            bg_color = "linear-gradient(90deg, #ff416c, #ff4b2b)" # Merah Menyala
            icon = "❌"
            label = "NEGATIF"
        else:
            bg_color = "linear-gradient(90deg, #bdc3c7, #2c3e50)" # Abu-abu
            icon = "😐"
            label = "NETRAL"

        # -- TAMPILAN HASIL (TABEL TENGAH) --
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color:white; margin-bottom:5px;">Hasil Prediksi</h4>
            <div style="font-style: italic; color: #a5b4fc; font-size: 14px; margin-bottom: 15px;">
                "{clean_text}"
            </div>
            <div class="sentiment-box" style="background: {bg_color};">
                {icon} {label}
            </div>
            <p style="margin-top: 15px; font-size: 12px; color: rgba(255,255,255,0.6);">
                Dianalisis menggunakan algoritma <b>{model_choice}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
