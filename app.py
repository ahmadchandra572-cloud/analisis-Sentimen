import streamlit as st
import joblib
import re
import string
import base64
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# 0️⃣ KONFIGURASI HALAMAN & ENCODER
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# --- FUNGSI UTILITY SINKRONISASI ---
def update_input_from_selectbox_asli():
    selected_value = st.session_state.selected_sample_asli
    if selected_value != "-- PILIH DATASET ASLI --":
        # Masuk otomatis ke bagian ketik sendiri
        st.session_state.current_input_area = selected_value 
        st.session_state.selected_sample_baku = "-- PILIH DATASET BAKU --"

def update_input_from_selectbox_baku():
    selected_value = st.session_state.selected_sample_baku
    if selected_value != "-- PILIH DATASET BAKU --":
        # Masuk otomatis ke bagian ketik sendiri
        st.session_state.current_input_area = selected_value
        st.session_state.selected_sample_asli = "-- PILIH DATASET ASLI --"

# --- INISIALISASI SESSION STATE ---
if 'current_input_area' not in st.session_state:
    st.session_state.current_input_area = ""
# ----------------------------------


def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Gambar Utama (Background) dan Gambar Tambahan
BG_IMAGE_FILENAME = "gamabr"
EXTRA_IMAGE_FILENAME = "images.jpg"

BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)


# ==========================================
# 1️⃣ CSS STYLE INJECTION
# ==========================================
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)),
            repeating-linear-gradient(
                45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px
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
        background-image: repeating-linear-gradient(45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px),
            radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }

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
    margin-bottom: 5px;
}

.subtitle { text-align: center; color: #8892b0; font-size: 1.1rem; margin-bottom: 2rem; }

.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}

.result-container { display: flex; justify-content: center; margin-top: 30px; }

.result-card {
    background: rgba(17, 34, 64, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px 25px;
    width: 100%;
    max-width: 550px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.sentiment-badge {
    font-size: 28px;
    font-weight: 700;
    padding: 10px 40px;
    border-radius: 50px;
    display: inline-block;
    color: white;
    margin-bottom: 20px;
}

/* Style Tabel Hasil */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 14px;
    text-align: left;
    background: rgba(0, 0, 0, 0.2);
}
.styled-table th, .styled-table td {
    padding: 12px 15px;
    border-bottom: 1px solid rgba(100, 255, 218, 0.1);
}
.styled-table th { color: #64ffda; text-transform: uppercase; letter-spacing: 1px; }
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
CHOSEN_MODEL_NAME = "Random Forest"
MODEL_TO_USE = MODELS[CHOSEN_MODEL_NAME] if MODELS and CHOSEN_MODEL_NAME in MODELS else None


# ==========================================
# 3.5️⃣ DAFTAR DATASET LENGKAP
# ==========================================
SAMPLE_COMMENTS_ASLI = [
    'Dpr jancok dpr tidak adil dasar', 'Setuju gaji anggota dewan umr supaya orang tidak ambisisius',
    'Brukakaka 1000% bayar PBB. Yang tinggal di kolong jembatan layak gk.', 'Mantap tarian jogetnya. Macam monyet dapat pisang.',
    'Apa dpr . Mau jaga rakyat . Atau mau siksa rakyat .', 'Kadang memang bikin hati panas dan ngerasa nggak adil.',
    'Semoga dapet musibah gaji dpr naik udh enak ksh tunjangan hidup', 'Pejabat paling terkorup indonesia.',
    'Puan bau tanah', 'Pantasan Rakyat pada marah kaya gini 😭😭😭'
]

SAMPLE_COMMENTS_BAKU = [
    'Kebijakan kenaikan tunjangan anggota DPR harus mempertimbangkan kondisi ekonomi masyarakat.',
    'Transparansi anggaran dalam pengalokasian dana fasilitas perumahan anggota dewan sangat diperlukan.',
    'Seharusnya pemerintah lebih mengutamakan peningkatan kesejahteraan guru honorer.',
    'Masyarakat menaruh harapan besar agar anggota DPR menolak fasilitas mewah.',
    'Integritas wakil rakyat diuji melalui keberanian mereka dalam membatasi pengeluaran anggaran.',
    'Semoga setiap keputusan yang diambil di gedung dewan selalu mendapatkan rida dari Tuhan Yang Maha Esa.'
]


# ==========================================
# 4️⃣ FUNGSI KOREKSI MANUAL
# ==========================================
def force_correct_prediction(clean_text: str, prediction: str) -> str:
    STRONG_NEGATIVE_KEYWORDS = ['buruk', 'jelek', 'bobrok', 'korup', 'bobrol', 'salah', 'tolak', 'gagal', 'miskin']
    if prediction.lower() == 'positif':
        if any(keyword in clean_text for keyword in STRONG_NEGATIVE_KEYWORDS):
            return 'Negatif (Koreksi Manual)'
    return prediction

# ==========================================
# 5️⃣ TAMPILAN UTAMA & LOGIKA PREDIKSI
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODEL_TO_USE is None:
    st.error("⚠️ Sistem gagal dimuat.")
    st.stop()

with st.container():
    col_img, col_input = st.columns([1, 2])
    with col_img:
        if EXTRA_IMAGE_B64:
            st.markdown('<div class="image-left-style">', unsafe_allow_html=True)
            st.image(f"data:image/jpeg;base64,{EXTRA_IMAGE_B64}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_input:
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>1. Pilih dari Dataset Asli:</p>", unsafe_allow_html=True)
        st.selectbox("Komentar Sampel Asli", options=["-- PILIH DATASET ASLI --"] + SAMPLE_COMMENTS_ASLI, key="selected_sample_asli", on_change=update_input_from_selectbox_asli, label_visibility="collapsed")
        
        st.markdown("<p style='font-weight: 600; margin-top: 10px; margin-bottom: 5px;'>2. Pilih dari Dataset Baku:</p>", unsafe_allow_html=True)
        st.selectbox("Komentar Sampel Baku", options=["-- PILIH DATASET BAKU --"] + SAMPLE_COMMENTS_BAKU, key="selected_sample_baku", on_change=update_input_from_selectbox_baku, label_visibility="collapsed")

        st.markdown("<p style='font-weight: 600; margin-top: 15px; margin-bottom: 5px;'>Atau Ketik Sendiri:</p>", unsafe_allow_html=True)
        # Text area sekarang menggunakan value dari session state untuk sinkronisasi otomatis
        input_text = st.text_area("Ketik Komentar", value=st.session_state.current_input_area, placeholder="Ketik di sini...", height=100, key="current_input_area", label_visibility="collapsed")
        analyze_button = st.button("🔍 ANALISIS SEKARANG")

if analyze_button:
    if st.session_state.current_input_area.strip() == "":
        st.warning("⚠️ Masukkan komentar!")
    else:
        # Proses Prediksi
        clean_text = text_preprocessing(st.session_state.current_input_area)
        X = VECTORIZER.transform([clean_text])
        
        # 1. PERHITUNGAN PROBABILITAS
        probs = MODEL_TO_USE.predict_proba(X)[0]
        max_prob = np.max(probs) * 100 
        
        # 2. LABEL ASLI & AKHIR
        ml_prediction = MODEL_TO_USE.predict(X)[0]
        final_prediction = force_correct_prediction(clean_text, ml_prediction)
        
        # Styling Badge
        if 'koreksi' in final_prediction.lower() or final_prediction.lower() == "negatif":
            label, badge_bg, icon = "NEGATIF", "linear-gradient(90deg, #dc2626, #f87171)", "😡"
        elif final_prediction.lower() == "positif":
            label, badge_bg, icon = "POSITIF", "linear-gradient(90deg, #059669, #34d399)", "😊"
        else:
            label, badge_bg, icon = "NETRAL", "linear-gradient(90deg, #64748b, #94a3b8)", "😐"

        # Tampilkan Hasil dalam Tabel
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <h4 style="color: #ccd6f6; margin-bottom: 15px;">HASIL ANALISIS SENTIMEN</h4>
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Keterangan Hasil</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Algoritma</td>
                            <td style="color: #64ffda;">{CHOSEN_MODEL_NAME} + GAM-GWO</td>
                        </tr>
                        <tr>
                            <td>Prediksi Murni Model</td>
                            <td>{ml_prediction.upper()}</td>
                        </tr>
                        <tr>
                            <td>Skor Confidence</td>
                            <td style="color: #64ffda;">{max_prob:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Status Akhir</td>
                            <td>{final_prediction}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
