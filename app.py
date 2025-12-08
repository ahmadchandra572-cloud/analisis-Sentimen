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
# 1️⃣ FUNGSI BACKGROUND & GAYA (CSS DARK NAVY)
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Nama file gambar (pastikan ada di repo)
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- WARNA TEMA: BIRU DONGKER (DARK NAVY) ---
# Kode warna: #0a192f (Deep Navy) sampai #112240 (Light Navy)

if BG_IMAGE_B64:
    # Gambar dengan overlay Biru Dongker Gelap
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
    # Fallback warna solid Biru Dongker jika gambar tidak ada
    background_css = """
    <style>
    .stApp {
        background: radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

ui_style = """
<style>
/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #ccd6f6;
}

/* Container Utama: Glassmorphism (Transparan Pudar) */
.block-container {
    background-color: rgba(17, 34, 64, 0.4); /* Biru dongker sangat transparan */
    backdrop-filter: blur(8px); /* Efek kaca buram */
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.1); /* Border tipis neon */
    box-shadow: 0 10px 30px -10px rgba(2, 12, 27, 0.7);
    max-width: 680px;
}

/* Header */
h1 {
    color: #e6f1ff;
    font-weight: 700;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #8892b0;
    font-size: 13px;
    margin-bottom: 25px;
    font-weight: 300;
}

/* Input Area (Textarea) */
.stTextArea textarea {
    background-color: rgba(10, 25, 47, 0.6); /* Gelap transparan */
    color: #e6f1ff;
    border: 1px solid #233554;
    border-radius: 10px;
}
.stTextArea textarea:focus {
    border-color: #64ffda; /* Hijau neon khas coding */
    box-shadow: 0 0 10px rgba(100, 255, 218, 0.1);
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(10, 25, 47, 0.6);
    color: white;
    border: 1px solid #233554;
    border-radius: 10px;
}

/* Tombol Analisis */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #112240, #233554);
    color: #64ffda; /* Teks Neon */
    border: 1px solid #64ffda;
    padding: 12px 0;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: 0.3s;
    margin-top: 15px;
}
.stButton > button:hover {
    background: rgba(100, 255, 218, 0.1);
    border-color: #64ffda;
    transform: translateY(-2px);
    color: white;
}

/* --- KARTU HASIL (PUDAR & MENYATU) --- */
.result-container {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

.result-card {
    background: rgba(17, 34, 64, 0.5); /* Sangat transparan (Pudar) */
    backdrop-filter: blur(10px); /* Blur kuat */
    border-radius: 16px;
    padding: 25px;
    width: 100%;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 10px 30px -15px rgba(2, 12, 27, 0.7);
    animation: fadeIn 0.8s ease-out;
}

.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8892b0;
    margin-bottom: 15px;
}

.sentiment-badge {
    display: inline-block;
    padding: 10px 40px;
    border-radius: 50px;
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.1);
}

.clean-text-box {
    background: rgba(2, 12, 27, 0.4);
    padding: 15px;
    border-radius: 8px;
    font-size: 13px;
    color: #a8b2d1;
    font-family: 'Courier New', monospace;
    margin-bottom: 10px;
    border-left: 2px solid #64ffda;
}

.model-info {
    font-size: 10px;
    color: #556080;
    margin-top: 15px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

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
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.error("⚠️ Sistem gagal dimuat.")
    st.stop()

# Layout Input
with st.container():
    model_choice = st.selectbox("Pilih Algoritma", list(MODELS.keys()))
    input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=100)
    analyze_button = st.button("ANALISIS SEKARANG")

# Logika Hasil
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # Proses
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Warna Badge (Tetap elegan tapi mencolok di background gelap)
        if prediction.lower() == "positif":
            badge_bg = "linear-gradient(90deg, #064e3b, #10b981)" # Hijau gelap ke terang
            icon = "😊"
            label = "POSITIF"
        elif prediction.lower() == "negatif":
            badge_bg = "linear-gradient(90deg, #7f1d1d, #ef4444)" # Merah gelap ke terang
            icon = "😡"
            label = "NEGATIF"
        else:
            badge_bg = "linear-gradient(90deg, #1e293b, #475569)" # Abu-abu
            icon = "😐"
            label = "NETRAL"

        # Tampilkan Kartu Hasil (Tengah & Pudar)
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
                
                <div class="model-info">
                    Algoritma: <b>{model_choice}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
