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
# 1️⃣ FUNGSI & STYLE CSS
# ==========================================
def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- LOAD GAMBAR ---
BG_IMAGE_FILENAME = "gamabr.jpg" # Pastikan nama file di GitHub sama persis!
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)

# --- DEFINISI CSS ---
# Kita gunakan logika: Jika gambar ada, pakai gambar. Jika tidak, pakai gradient.
if BG_IMAGE_B64:
    # OPSI A: BACKGROUND GAMBAR (Dengan Overlay Biru Gelap biar teks terbaca)
    background_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(15, 23, 42, 0.80), rgba(15, 23, 42, 0.90)), 
                          url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    # OPSI B: GRADIENT CADANGAN (Jika gambar error/hilang)
    background_css = """
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        background-attachment: fixed;
    }
    """

# --- INJECT CSS ---
st.markdown(f"""
<style>
    /* Aplikasi Background dari Logika di atas */
    {background_css}

    /* Styling Container Utama (Kotak Kaca/Glassmorphism) */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        max-width: 700px;
    }}

    /* Header Styling */
    h1 {{
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica', sans-serif;
        font-weight: 800;
        text-align: center;
        padding-bottom: 10px;
    }}
    
    /* Text Color Override */
    h3, p, label, .stMarkdown {{
        color: #e0e7ff !important;
    }}

    /* Input Area Styling */
    .stTextArea textarea {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #4b5563;
    }}
    .stTextArea textarea:focus {{
        border: 1px solid #00C9FF;
        box-shadow: 0 0 10px rgba(0, 201, 255, 0.3);
    }}

    /* Selectbox Styling */
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: rgba(0, 0, 0, 0.4);
        color: white;
        border-radius: 10px;
        border: 1px solid #4b5563;
    }}

    /* Button Styling */
    .stButton button {{
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }}
    .stButton button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }}

    /* --- HASIL PREDIKSI (RESULT CARD) --- */
    .result-card {{
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 25px;
        margin-top: 30px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        animation: fadeIn 0.8s ease-out;
    }}
    
    .sentiment-label {{
        font-size: 28px;
        font-weight: 800;
        margin: 15px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
    }}

    /* Animasi */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2️⃣ LOGIKA SISTEM (LOAD & PREPROCESS)
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
# 3️⃣ USER INTERFACE (LAYOUT)
# ==========================================

# -- Header --
st.markdown("<h1>📊 ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Deteksi Opini Publik tentang Gaji DPR menggunakan Optimasi GAM-GWO</p>", unsafe_allow_html=True)

if VECTORIZER is None or MODELS is None:
    st.stop()

# -- Form Input --
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("🤖 Pilih Model", list(MODELS.keys()))
    with col2:
        st.write("") # Spacer agar sejajar

    input_text = st.text_area("✍️ Masukkan komentar di sini...", height=100, placeholder="Contoh: Kinerja DPR harus ditingkatkan...")

    # Tombol Analisis (Tengah)
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        analyze_button = st.button("🚀 ANALISIS SEKARANG", use_container_width=True)

# -- Hasil Analisis --
if analyze_button:
    if input_text.strip() == "":
        st.warning("⚠️ Harap masukkan teks terlebih dahulu!")
    else:
        # Proses Prediksi
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]
        
        # Penentuan Warna & Ikon
        if prediction.lower() == "positif":
            color_css = "background: linear-gradient(90deg, #00b09b, #96c93d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
            icon = "✅"
            status_text = "POSITIF"
            box_border = "#96c93d"
        elif prediction.lower() == "negatif":
            color_css = "background: linear-gradient(90deg, #ff416c, #ff4b2b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
            icon = "❌"
            status_text = "NEGATIF"
            box_border = "#ff4b2b"
        else:
            color_css = "background: linear-gradient(90deg, #bdc3c7, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
            icon = "😐"
            status_text = "NETRAL"
            box_border = "#bdc3c7"

        # -- MENAMPILKAN CARD HASIL --
        st.markdown(f"""
        <div class="result-card" style="border-top: 5px solid {box_border};">
            <p style="color: #cbd5e1; font-size: 14px; margin-bottom: 5px;">Teks Terproses:</p>
            <div style="font-style: italic; color: #a5b4fc; font-size: 16px; margin-bottom: 20px; font-weight: 500;">
                "{clean_text}"
            </div>
            
            <div style="border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;"></div>
            
            <p style="color: white; font-size: 14px; margin-top: 15px;">Prediksi Sentimen:</p>
            <div class="sentiment-label">
                {icon} <span style="{color_css}">{status_text}</span>
            </div>
            
            <p style="margin-top: 15px; font-size: 11px; color: rgba(255,255,255,0.5);">
                Dianalisis oleh: <b>{model_choice}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
