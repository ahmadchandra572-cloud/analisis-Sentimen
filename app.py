import streamlit as st
import joblib
import re
import string
import base64
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 


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
        # st.error(f"File not found: {file_path}") # Jangan tampilkan error di awal
        return None

# Gambar Utama (Background) dan Gambar Tambahan
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
EXTRA_IMAGE_FILENAME = "images.jpg"
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)


# ==========================================
# 1️⃣ CSS STYLE INJECTION (Minimalis, Fokus, Background Lebih Jelas)
# ==========================================
# --- Background Image + Overlay ---
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image: 
            /* Layer 2: Overlay Biru Dongker (40%-60% opacity) */
            linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)), 
            /* Layer 3: Pola Grid Halus */
            repeating-linear-gradient(
                45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px
            ),
            /* Base Image */
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
        background-image: repeating-linear-gradient(45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px),
            radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

# --- UI Styling (Fokus pada Badge) ---
ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }

/* Container Utama (Glassmorphism) */
.block-container {
    background-color: rgba(17, 34, 64, 0.2); 
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.08); 
    max-width: 900px; /* Melebarkan untuk kolom gambar */
}

h1 {
    font-weight: 700;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

/* Gambar Kiri */
.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}

.result-container { display: flex; justify-content: center; margin-top: 30px; }

/* Kartu Hasil Sederhana */
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

.sentiment-badge { 
    font-size: 28px; 
    font-weight: 700; 
    padding: 15px 40px; 
    border-radius: 50px; 
    display: inline-block; 
    color: white; 
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Styling untuk Expander Kata Kunci */
.st-emotion-cache-p5m8c2 { /* Target Streamlit expander header */
    background: rgba(100, 255, 218, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px !important;
    border: none !important;
}
.sentiment-example-box {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 5px;
    font-size: 13px;
    color: #e0e7ff;
}
.positive { background-color: rgba(52, 211, 153, 0.1); border-left: 4px solid #34d399; }
.negative { background-color: rgba(248, 113, 113, 0.1); border-left: 4px solid #f87171; }
.neutral { background-color: rgba(100, 116, 139, 0.1); border-left: 4px solid #94a3b8; }

/* Gaya untuk Algoritma yang Dipilih */
.chosen-algo {
    font-size: 16px;
    font-weight: 600;
    color: #64ffda;
    background-color: rgba(100, 255, 218, 0.05);
    padding: 8px 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid rgba(100, 255, 218, 0.2);
}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)


# ==========================================
# 2️⃣ PREPROCESSING & RESOURCE LOADING
# ==========================================
try:
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
        st.error(f"Gagal memuat sistem. Pastikan file 'tfidf_vectorizer.pkl', 'model_RF_GamGwo.pkl', 'model_LR_GamGwo.pkl', dan 'model_SVM_GamGwo.pkl' ada: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# 3️⃣ KONFIGURASI ALGORITMA SATU (GABUNGAN)
# ==========================================
# Kita pilih model terbaik (misalnya Random Forest) sebagai model utama.
# Ini mensimulasikan "gabungan" di mana hanya satu hasil terbaik yang digunakan.
CHOSEN_MODEL_NAME = "Random Forest"
if MODELS and CHOSEN_MODEL_NAME in MODELS:
    MODEL_TO_USE = MODELS[CHOSEN_MODEL_NAME]
else:
    MODEL_TO_USE = None


# ==========================================
# 4️⃣ TAMPILAN UTAMA & LOGIKA PREDIKSI
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODEL_TO_USE is None:
    st.error("⚠️ Sistem gagal dimuat atau model tidak ditemukan.")
    st.stop()

# --- BLOK CONTOH KATA (Panduan) ---
with st.expander("📚 Contoh Kata Kunci Sentimen (Panduan)", expanded=False):
    col_pos, col_neg, col_net = st.columns(3)

    # Kata Kunci
    positive_words = ["mantap", "bagus", "sukses", "hebat", "terbaik", "cocok", "adil", "bijak", "bersyukur"]
    negative_words = ["tolak", "gagal", "rugi", "miskin", "korupsi", "mahal", "bodoh", "malu", "kecewa"]
    neutral_words = ["rapat", "usulan", "pimpinan", "komisi", "kebijakan", "anggaran", "membahas", "jakarta", "sidang"]

    with col_pos:
        st.markdown("<h4 style='color: #34d399;'>POSITIF</h4>", unsafe_allow_html=True)
        for word in positive_words:
            st.markdown(f'<div class="sentiment-example-box positive">✅ {word}</div>', unsafe_allow_html=True)
    
    with col_neg:
        st.markdown("<h4 style='color: #f87171;'>NEGATIF</h4>", unsafe_allow_html=True)
        for word in negative_words:
            st.markdown(f'<div class="sentiment-example-box negative">❌ {word}</div>', unsafe_allow_html=True)
    
    with col_net:
        st.markdown("<h4 style='color: #94a3b8;'>NETRAL</h4>", unsafe_allow_html=True)
        for word in neutral_words:
            st.markdown(f'<div class="sentiment-example-box neutral">⚪ {word}</div>', unsafe_allow_html=True)
# ---------------------------------------------


# Layout Input
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
        # Pilihan Algoritma dihapus dari sini.
        
        input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=100)
        analyze_button = st.button("🔍 ANALISIS SEKARANG")

# Logika Hasil
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # Proses Prediksi
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        
        # Model yang digunakan adalah model yang sudah ditetapkan di awal (MODEL_TO_USE)
        prediction = MODEL_TO_USE.predict(X)[0]
        
        # Styling Hasil
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

        # Tampilkan Kartu Hasil (Minimalis - Hanya Badge)
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <h4 style="color: #ccd6f6; margin-bottom: 5px;">HASIL ANALISIS SENTIMEN</h4>
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                <small style='color: #94a3b8;'>Menggunakan model {CHOSEN_MODEL_NAME} (Gabungan Hasil Terbaik)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
