import streamlit as st
import joblib
import re
import string
import base64
# Sastrawi tidak perlu diimpor di sini, karena sudah ditangani di bagian try/except global di bawah

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

# NAMA FILE GAMBAR
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- STYLE CSS (Biru Dongker Glassmorphism) ---
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    /* Background Image dengan overlay */
    .stApp {{
        background-image: linear-gradient(rgba(10, 25, 47, 0.90), rgba(10, 25, 47, 0.95)), 
                          url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
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
/* Font dan Layout Umum */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }

/* Container Utama (Glassmorphism) */
.block-container {
    background-color: rgba(17, 34, 64, 0.4); 
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.1); 
    max-width: 680px;
}

/* Judul dan Sub-teks */
h1 {
    font-weight: 700;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 5px;
    letter-spacing: 1px;
}
.subtitle {
    text-align: center;
    color: #cbd6f6;
    font-weight: 300;
    margin-bottom: 25px;
}

/* Kartu Hasil yang Centered */
.result-card {
    background: rgba(17, 34, 64, 0.5); 
    backdrop-filter: blur(10px); 
    border-radius: 16px;
    padding: 25px;
    width: 100%;
    text-align: center; /* Memastikan isinya di tengah */
}

.sentiment-badge { 
    font-size: 24px; font-weight: 700; padding: 12px 35px; border-radius: 50px; 
    display: inline-block; color: white; margin-bottom: 20px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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

.model-info {
    font-size: 10px;
    color: #556080;
    margin-top: 15px;
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
st.markdown("<h1>ANALISIS SENTIMEN ML</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.error("⚠️ Sistem gagal dimuat.")
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
        # Proses Prediksi
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        tfidf_shape = X.shape # Menampilkan bentuk vektor (TF-IDF)
        
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
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

        # Tampilkan Kartu Hasil (Fokus pada Labeling Data ML)
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <div class="result-label">HASIL PELABELAN DATA (ML)</div>
                
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                
                <div style="text-align: left; margin-bottom: 5px; font-size: 11px; color: #8892b0; margin-left: 5px;">
                    Teks Bersih (Input untuk ML):
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
                    Algoritma yang Digunakan: <b>{model_choice}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
