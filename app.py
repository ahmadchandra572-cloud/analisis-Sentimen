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
# 1️⃣ FUNGSI BACKGROUND & GAYA (CSS)
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# PENTING: Sesuaikan nama ini dengan nama file di GitHub Anda
# Berdasarkan gambar Anda sebelumnya, nama filenya adalah 'gamabr' (tanpa .jpg)
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- CSS STYLING ---
# Menyiapkan background
if BG_IMAGE_B64:
    # Jika gambar ketemu, pakai gambar + overlay biru gelap
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(15, 23, 42, 0.80), rgba(15, 23, 42, 0.90)), 
                          url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
else:
    # Jika gambar tidak ketemu, pakai warna solid biru gelap
    page_bg_img = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }
    </style>
    """

# Menyiapkan Style UI (Kartu, Header, Tombol)
ui_style = """
<style>
/* Container Utama Transparan */
.block-container {
    background-color: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 3rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    max-width: 700px;
}

/* Header Text */
h1 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    text-align: center;
    padding-bottom: 10px;
}

h3, p, label, .stMarkdown {
    color: #e2e8f0 !important;
}

/* Text Area Input */
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.6);
    color: white;
    border: 1px solid #475569;
    border-radius: 12px;
}

/* Tombol Analisis */
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: bold;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
}

/* Kartu Hasil (Result Card) */
.result-card {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
}

.sentiment-badge {
    font-size: 28px;
    font-weight: 800;
    padding: 12px 30px;
    border-radius: 50px;
    display: inline-block;
    margin: 20px 0;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

# Terapkan CSS
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)


# ==========================================
# 2️⃣ LOAD DEPENDENCIES (SASTRAWI & RESOURCES)
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
st.markdown("<p style='text-align: center; opacity: 0.7; margin-top: -15px;'>Deteksi Opini Publik Isu Gaji DPR dengan Optimasi GAM-GWO</p>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.warning("⚠️ Sistem sedang memuat model... Jika lama, coba refresh.")
    st.stop()

# Layout Input
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("⚙️ Pilih Algoritma", list(MODELS.keys()))
    with col2:
        st.write("") # Spacer

    input_text = st.text_area("", placeholder="Ketik komentar di sini...", height=100)

    col_l, col_m, col_r = st.columns([1, 1.5, 1])
    with col_m:
        analyze_button = st.button("🔍 ANALISIS SEKARANG", use_container_width=True)

# Logika Hasil
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # Proses Prediksi
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Styling Hasil
        if prediction.lower() == "positif":
            badge_color = "linear-gradient(135deg, #059669, #34d399)"
            icon = "😊"
            label = "POSITIF"
        elif prediction.lower() == "negatif":
            badge_color = "linear-gradient(135deg, #dc2626, #f87171)"
            icon = "😡"
            label = "NEGATIF"
        else:
            badge_color = "linear-gradient(135deg, #475569, #94a3b8)"
            icon = "😐"
            label = "NETRAL"

        # Tampilkan Kartu Hasil
        st.markdown(f"""
        <div class="result-card">
            <p style="color:#94a3b8; font-size:14px; letter-spacing:1px; text-transform:uppercase;">Hasil Prediksi</p>
            
            <div class="sentiment-badge" style="background: {badge_color};">
                {icon} &nbsp; {label}
            </div>
            
            <div style="background:rgba(0,0,0,0.2); padding:15px; border-radius:10px; text-align:left; margin-top:10px;">
                <p style="color:#cbd5e1; font-size:12px; margin:0;">Teks Asli:</p>
                <p style="color:white; font-style:italic; margin:5px 0;">"{input_text}"</p>
                <hr style="border-color:rgba(255,255,255,0.1);">
                <p style="color:#cbd5e1; font-size:12px; margin:0;">Teks Bersih (Preprocessed):</p>
                <p style="color:#60a5fa; font-family:monospace; margin:5px 0;">{clean_text}</p>
            </div>
            
            <p style="margin-top: 15px; font-size: 11px; opacity: 0.5;">
                Model: <b>{model_choice}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
