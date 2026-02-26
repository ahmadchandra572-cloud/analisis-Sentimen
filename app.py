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
    selected_item = st.session_state.selected_sample_asli
    if selected_item != "-- PILIH DATASET ASLI --":
        # Ambil teks komentarnya saja
        st.session_state.current_input_area = selected_item[0]
        # Simpan label aslinya ke session state
        st.session_state.original_label = selected_item[1]
        st.session_state.selected_sample_baku = "-- PILIH DATASET BAKU --"

def update_input_from_selectbox_baku():
    selected_item = st.session_state.selected_sample_baku
    if selected_item != "-- PILIH DATASET BAKU --":
        st.session_state.current_input_area = selected_item[0]
        st.session_state.original_label = selected_item[1]
        st.session_state.selected_sample_asli = "-- PILIH DATASET ASLI --"

# --- INISIALISASI SESSION STATE ---
if 'current_input_area' not in st.session_state:
    st.session_state.current_input_area = ""
if 'original_label' not in st.session_state:
    st.session_state.original_label = "-"

def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

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
        background-image: linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)), 
        url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
    }}
    </style>
    """
else:
    background_css = """<style>.stApp { background-color: #0a192f; }</style>"""

ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }
.block-container { background-color: rgba(17, 34, 64, 0.2); backdrop-filter: blur(12px); border-radius: 20px; padding: 3rem 2rem !important; border: 1px solid rgba(100, 255, 218, 0.08); }
h1 { font-weight: 700; background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
.subtitle { text-align: center; color: #8892b0; font-size: 1.1rem; margin-bottom: 2rem; }

.result-card-container {
    background: rgba(17, 34, 64, 0.6);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(100, 255, 218, 0.15);
    margin-top: 30px;
}
.sentiment-display { text-align: center; border-right: 1px solid rgba(100, 255, 218, 0.1); padding-right: 20px; }
.sentiment-badge { font-size: 30px; font-weight: 800; padding: 10px 25px; border-radius: 15px; color: white; display: block; margin-top: 10px; }
.explanation-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-left: 10px; }
.explanation-table td { padding: 8px 5px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
.label-text { color: #64ffda; font-weight: 600; width: 40%; }
.value-text { color: #ccd6f6; }
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
    if STEMMER: text = STEMMER.stem(text)
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
# 3️⃣ DATASET ASLI & BAKU DENGAN LABEL ASLI
# ==========================================
# Format: (Teks Komentar, Label Asli Dataset)

SAMPLE_DATA_ASLI = [
    # Data asli dari kode awal kamu
    ('Setuju gaji anggota dewan umr supaya orang tidak ambisisius', 'positif'),
    ('Brukakaka 1000% bayar PBB. Yang tinggal di kolong jembatan layak gk.', 'Negatif'),
    ('Mantap tarian jogetnya. Macam monyet dapat pisang.', 'Positif'),
    ('Apa dpr . Mau jaga rakyat . Atau mau siksa rakyat .', 'Negatif'),
    ('Pantasan Rakyat pada marah kaya gini 😭😭😭', 'Negatif'),
    
    # Tambahan variasi bahasa gaul/medsos (tidak baku)
    ('DPR naik gaji lagi, rakyat kok makin susah ya 😭', 'Negatif'),
    ('Enak banget jadi anggota dewan, gaji gede fasilitas mewah', 'Negatif'),
    ('Gaji DPR naik kok ga naik upah buruh? DPR pro rakyat mana?', 'Negatif'),
    ('Mantap lah, kerja sebulan dapet ratusan juta, enak tenan', 'Positif'),
    ('DPR cuma bisa bikin rakyat sengsara, bubarin aja', 'Negatif'),
    ('Setuju banget gaji DPR dinaikin, biar lebih semangat kerja', 'Positif'),
    ('Pantesan DPR cuek sama rakyat, gajinya udah kayak sultan', 'Negatif'),
    ('Gaji kecil katanya, makanya korupsi mulu wkwkwk', 'Negatif'),
    ('Terima kasih DPR sudah perjuangkan kesejahteraan rakyat 🙏', 'Positif'),
    ('Naikin tunjangan lagi? mending dana itu buat bansos', 'Negatif'),
    ('DPR kok gak pernah mikirin nasib rakyat kecil ya?', 'Negatif'),
    ('Bagus dong gaji naik, biar ga tergoda korupsi lagi', 'Positif'),
]

SAMPLE_DATA_BAKU = [
    # Data asli dari kode awal kamu
    ('Kebijakan kenaikan tunjangan anggota DPR harus mempertimbangkan kondisi ekonomi masyarakat.', 'Netral'),
    ('Transparansi anggaran dalam pengalokasian dana fasilitas perumahan anggota dewan sangat diperlukan.', 'Netral'),
    ('Seharusnya pemerintah lebih mengutamakan peningkatan kesejahteraan guru honorer.', 'Positif'),
    ('Masyarakat menaruh harapan besar agar anggota DPR menolak fasilitas mewah.', 'Positif'),
    ('Semoga setiap keputusan yang diambil di gedung dewan selalu mendapatkan rida dari Tuhan Yang Maha Esa.', 'Positif'),
    
    # Tambahan variasi bahasa formal/baku
    ('Kenaikan gaji anggota DPR perlu disertai evaluasi kinerja yang transparan dan akuntabel.', 'Netral'),
    ('Pemerintah disarankan untuk memprioritaskan anggaran pendidikan dan kesehatan di atas tunjangan legislatif.', 'Negatif'),
    ('Peningkatan fasilitas perumahan bagi anggota dewan sebaiknya dipertimbangkan kembali mengingat kondisi fiskal negara.', 'Netral'),
    ('Anggota DPR diharapkan dapat menunjukkan empati yang lebih besar terhadap kesulitan ekonomi masyarakat.', 'Negatif'),
    ('Kebijakan remunerasi legislatif harus sejalan dengan prinsip keadilan sosial dan pemerataan kesejahteraan.', 'Netral'),
    ('Peningkatan kesejahteraan anggota dewan dapat memotivasi kinerja yang lebih baik dalam legislasi.', 'Positif'),
    ('Perlu adanya mekanisme pengawasan publik terhadap penggunaan anggaran fasilitas anggota DPR.', 'Netral'),
    ('Masyarakat berharap DPR lebih fokus pada pembahasan RUU yang pro-rakyat daripada kesejahteraan pribadi.', 'Negatif'),
    ('Kenaikan tunjangan sebaiknya diiringi dengan peningkatan tanggung jawab dan integritas anggota dewan.', 'Netral'),
    ('DPR memiliki peran strategis dalam memperjuangkan kesejahteraan rakyat secara keseluruhan.', 'Positif'),
    ('Anggaran negara seharusnya dialokasikan secara proporsional untuk kepentingan mayoritas rakyat.', 'Netral'),
    ('Peningkatan fasilitas kerja anggota dewan perlu dibarengi dengan peningkatan pelayanan publik yang nyata.', 'Netral'),
]

def force_correct_prediction(clean_text: str, prediction: str) -> str:
    STRONG_NEG = ['buruk', 'jelek', 'bobrok', 'korup', 'bobrol', 'salah', 'tolak', 'gagal', 'miskin']
    if prediction.lower() == 'positif':
        if any(keyword in clean_text for keyword in STRONG_NEG):
            return 'Negatif (Koreksi Lexicon)'
    return prediction

# ==========================================
# 5️⃣ TAMPILAN UTAMA
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

    with col_input:
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>1. Pilih dari Dataset Asli:</p>", unsafe_allow_html=True)
        st.selectbox("Sampel Asli", options=["-- PILIH DATASET ASLI --"] + SAMPLE_DATA_ASLI, 
                     format_func=lambda x: x[0] if isinstance(x, tuple) else x,
                     key="selected_sample_asli", on_change=update_input_from_selectbox_asli, label_visibility="collapsed")
        
        st.markdown("<p style='font-weight: 600; margin-top: 10px; margin-bottom: 5px;'>2. Pilih dari Dataset Baku:</p>", unsafe_allow_html=True)
        st.selectbox("Sampel Baku", options=["-- PILIH DATASET BAKU --"] + SAMPLE_DATA_BAKU, 
                     format_func=lambda x: x[0] if isinstance(x, tuple) else x,
                     key="selected_sample_baku", on_change=update_input_from_selectbox_baku, label_visibility="collapsed")

        st.markdown("<p style='font-weight: 600; margin-top: 10px; margin-bottom: 5px;'>3. Pilih Algoritma Terbaik:</p>", unsafe_allow_html=True)
        chosen_algo = st.selectbox("Pilih Model", options=["Random Forest", "Logistic Regression", "SVM"], key="selected_algo", label_visibility="collapsed")

        st.markdown("<p style='font-weight: 600; margin-top: 15px; margin-bottom: 5px;'>Input Komentar (Otomatis Terisi):</p>", unsafe_allow_html=True)
        input_text = st.text_area("Ketik Komentar", value=st.session_state.current_input_area, height=80, key="current_input_area", label_visibility="collapsed")
        
        analyze_button = st.button("🔍 ANALISIS SEKARANG")

if analyze_button:
    if st.session_state.current_input_area.strip() == "":
        st.warning("⚠️ Masukkan komentar!")
    else:
        MODEL_TO_USE = MODELS[chosen_algo]
        clean_text = text_preprocessing(st.session_state.current_input_area)
        X = VECTORIZER.transform([clean_text])
        
        ml_prediction = MODEL_TO_USE.predict(X)[0]
        final_prediction = force_correct_prediction(clean_text, ml_prediction)
        try:
            probs = MODEL_TO_USE.predict_proba(X)[0]
            max_prob = f"{np.max(probs) * 100:.2f}%"
        except: max_prob = "100.00%"

        if 'negatif' in final_prediction.lower(): label, badge_bg, icon = "NEGATIF", "linear-gradient(135deg, #dc2626, #b91c1c)", "😡"
        elif 'positif' in final_prediction.lower(): label, badge_bg, icon = "POSITIF", "linear-gradient(135deg, #059669, #047857)", "😊"
        else: label, badge_bg, icon = "NETRAL", "linear-gradient(135deg, #4b5563, #374151)", "😐"

        # Tampilan 2 Kolom
        st.markdown(f"""
        <div class="result-card-container">
            <div style="display: flex; align-items: center;">
                <div style="flex: 1;" class="sentiment-display">
                    <p style="margin-bottom: 0; font-size: 14px; color: #8892b0;">HASIL AKHIR</p>
                    <div class="sentiment-badge" style="background: {badge_bg};">
                        {icon} <br> {label}
                    </div>
                </div>
                <div style="flex: 2; padding-left: 20px;">
                    <table class="explanation-table">
                        <tr><td class="label-text">🤖 Algoritma</td><td class="value-text">{chosen_algo} + GAM-GWO</td></tr>
                        <tr><td class="label-text">📂 Label Asli Teks</td><td class="value-text" style="color: #64ffda; font-weight: bold;">{st.session_state.original_label.upper()}</td></tr>
                        <tr><td class="label-text">📊 Prediksi Model</td><td class="value-text">Model memprediksi sebagai <b>{ml_prediction.upper()}</b>.</td></tr>
                        <tr><td class="label-text">📈 Confidence</td><td class="value-text">Tingkat keyakinan: <b>{max_prob}</b>.</td></tr>
                        <tr><td class="label-text">📝 Status Akhir</td><td class="value-text" style="color: #64ffda;">{final_prediction}</td></tr>
                    </table>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
