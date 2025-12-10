import streamlit as st
import joblib
import re
import string
import base64
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import pandas as pd
from typing import Dict, List


# ==========================================
# 0️⃣ KONFIGURASI HALAMAN & ENCODER
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# --- DEFINISI KATA KUNCI SENTIMEN (Untuk Digunakan dalam Logika) ---
SENTIMENT_KEYWORDS: Dict[str, List[str]] = {
    "positif": ["mantap", "bagus", "sukses", "hebat", "terbaik", "cocok", "adil", "bijak", "bersyukur", "setuju", "dukung", "layak", "profesional"],
    "negatif": ["tolak", "gagal", "rugi", "miskin", "korupsi", "mahal", "bodoh", "malu", "kecewa", "bubar", "ancur", "menyesal", "gila", "membabi buta", "memeras", "bobrok"],
    "netral": ["rapat", "usulan", "pimpinan", "komisi", "kebijakan", "anggaran", "membahas", "jakarta", "sidang", "peraturan", "kabar", "berita", "isu"]
}

def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Gambar Utama (Background) dan Gambar Tambahan
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
EXTRA_IMAGE_FILENAME = "images.jpg"
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)


# ==========================================
# 1️⃣ CSS STYLE INJECTION (Minimalis, Fokus, Background Lebih Jelas)
# ==========================================
# ... (CSS Styling Tetap Sama) ...
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
    """ if BG_IMAGE_B64 else """
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

/* Container Utama (Glassmorphism) */
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

.result-container { display: flex; flex-direction: column; align-items: center; margin-top: 30px; }

/* Kartu Hasil yang Diperluas */
.result-card {
    background: rgba(17, 34, 64, 0.5); 
    backdrop-filter: blur(10px); 
    border-radius: 16px;
    padding: 25px 30px; 
    width: 100%;
    max-width: 700px; 
    text-align: left; /* Diubah agar teks penjelasan rata kiri */
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.result-card h4 { text-align: center; }

.sentiment-badge { 
    font-size: 28px; 
    font-weight: 700; 
    padding: 15px 40px; 
    border-radius: 50px; 
    display: inline-block; 
    color: white; 
    margin: 10px auto 20px auto; /* auto agar di tengah */
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Styling untuk penyorotan kata kunci */
.highlight-pos { background-color: rgba(52, 211, 153, 0.4); padding: 2px 4px; border-radius: 3px; font-weight: 600; }
.highlight-neg { background-color: rgba(248, 113, 113, 0.4); padding: 2px 4px; border-radius: 3px; font-weight: 600; }
.highlight-net { background-color: rgba(100, 116, 139, 0.4); padding: 2px 4px; border-radius: 3px; font-weight: 600; }
.analysis-text { margin-top: 15px; line-height: 1.8; font-size: 14px; text-align: justify; }

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
    # text = text.encode('ascii', 'ignore').decode('ascii') # Baris ini dihapus agar karakter non-ascii Indonesia tetap ada
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
        st.error(f"Gagal memuat sistem. Pastikan file model dan vectorizer ada: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# 3️⃣ KONFIGURASI ALGORITMA SATU (GABUNGAN)
# ==========================================
# Kita pilih model terbaik (misalnya Random Forest) sebagai model utama.
CHOSEN_MODEL_NAME = "Random Forest"
if MODELS and CHOSEN_MODEL_NAME in MODELS:
    MODEL_TO_USE = MODELS[CHOSEN_MODEL_NAME]
else:
    MODEL_TO_USE = None


# ==========================================
# 4️⃣ FUNGSI UTILITY & PENJELASAN
# ==========================================

def highlight_keywords(text: str, keywords: Dict[str, List[str]]) -> str:
    """Menyorot kata kunci sentimen dalam teks asli (belum distemming)."""
    
    # Gabungkan semua keyword untuk efisiensi
    all_keywords = {}
    for sentiment, words in keywords.items():
        for word in words:
            all_keywords[word.lower()] = sentiment

    # Tokenisasi teks untuk mencari kecocokan yang tepat (menggunakan regex untuk word boundary)
    highlighted_text = text
    for word, sentiment in all_keywords.items():
        # Membuat regex yang mencari kata lengkap, mengabaikan huruf besar/kecil
        pattern = r'\b(' + re.escape(word) + r')\b'
        
        # Tentukan kelas CSS
        css_class = ""
        if sentiment == "positif":
            css_class = "highlight-pos"
        elif sentiment == "negatif":
            css_class = "highlight-neg"
        elif sentiment == "netral":
            css_class = "highlight-net"

        # Ganti kata yang cocok dengan tag span HTML
        highlighted_text = re.sub(
            pattern, 
            lambda m: f'<span class="{css_class}">{m.group(0)}</span>', 
            highlighted_text, 
            flags=re.IGNORECASE
        )
    return highlighted_text

def generate_long_explanation(sentiment: str, original_text: str, model_name: str) -> str:
    """
    Menghasilkan penjelasan yang panjang dan formatif berdasarkan hasil prediksi.
    (Panjang sekitar 300 kata)
    """
    
    # 1. Deteksi Kata Kunci
    detected_words = {}
    tokens = re.findall(r'\b\w+\b', original_text.lower())
    
    for sent, words in SENTIMENT_KEYWORDS.items():
        for word in words:
            if word in tokens:
                if sent not in detected_words:
                    detected_words[sent] = []
                detected_words[sent].append(word)

    # 2. Persiapan Teks Penjelasan Utama
    
    # Header berdasarkan sentimen
    if sentiment.lower() == "positif":
        main_point = "Opini ini menunjukkan pandangan yang **mendukung** atau **mengapresiasi** isu terkait DPR. Sentimen positif seringkali terkait dengan harapan akan kinerja yang baik, hasil yang memuaskan, atau dukungan terhadap keputusan yang telah dibuat."
        color = "#34d399"
        summary_intro = f"Kesimpulan analisis sentimen terhadap komentar ini adalah **POSITIF**."
    elif sentiment.lower() == "negatif":
        main_point = "Komentar ini teridentifikasi sebagai **NEGATIF**, mencerminkan adanya **ketidakpuasan, kritik keras, atau penolakan** terhadap kebijakan atau kinerja anggota DPR, khususnya terkait isu gaji. Ini menunjukkan adanya keresahan publik yang signifikan."
        color = "#f87171"
        summary_intro = f"Kesimpulan analisis sentimen terhadap komentar ini adalah **NEGATIF**."
    else:
        main_point = "Sentimen yang terdeteksi adalah **NETRAL**. Teks ini cenderung menyajikan **fakta, informasi, atau pernyataan yang tidak emosional**, tanpa memberikan penilaian yang jelas baik pro maupun kontra terhadap subjek (isu gaji DPR)."
        color = "#94a3b8"
        summary_intro = f"Kesimpulan analisis sentimen terhadap komentar ini adalah **NETRAL**."

    # Bagian 3: Analisis Keyword
    keyword_analysis = ""
    total_detected = sum(len(v) for v in detected_words.values())
    
    if total_detected > 0:
        keyword_analysis += "<h5 style='color: #64ffda; margin-top: 15px; margin-bottom: 5px;'>Analisis Kata Kunci (Keywords)</h5>"
        keyword_analysis += "Sistem mendeteksi keberadaan kata-kata kunci berikut dalam teks Anda, yang turut memperkuat hasil prediksi sentimen:<ul>"
        
        for sent, words in detected_words.items():
            if words:
                color_map = {'positif': '#34d399', 'negatif': '#f87171', 'netral': '#94a3b8'}
                keyword_analysis += f"<li><strong style='color: {color_map.get(sent, '#ccd6f6')};'>{sent.upper()} ({len(words)} kata):</strong> {', '.join(words)}</li>"
        
        keyword_analysis += "</ul>"
        
    else:
        keyword_analysis += "<p>Teks tidak mengandung kata kunci sentimen yang spesifik dari kamus, sehingga prediksi sentimen lebih mengandalkan pola kontekstual yang dipelajari oleh model.</p>"

    # Bagian 4: Kesimpulan dan Metode
    conclusion = f"""
    <p>
    <strong>Dasar Klasifikasi:</strong> Komentar Anda diklasifikasikan menggunakan model <strong>{model_name}</strong> yang telah dioptimasi. Model ini bekerja dengan menganalisis pola kemunculan kata (menggunakan representasi TF-IDF) setelah proses pra-pemrosesan data yang meliputi *case folding*, penghapusan *stopword*, dan *stemming* bahasa Indonesia. 
    Akurasi klasifikasi ini berasal dari pemahaman model terhadap ribuan data komentar publik terkait isu yang sama. Keakuratan hasil ini mengindikasikan bahwa pola leksikal dan sintaksis dalam komentar Anda sangat mirip dengan pola yang ada pada data pelatihan sentimen {sentiment.upper()}.
    </p>
    """

    # Gabungkan semua bagian
    full_explanation = f"""
    <div style="padding: 15px 0; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <h4 style="color: {color}; margin-bottom: 10px;">{summary_intro}</h4>
        <p class="analysis-text">{main_point}</p>
        {keyword_analysis}
        {conclusion}
    </div>
    """
    
    return full_explanation.strip()

# ==========================================
# 5️⃣ TAMPILAN UTAMA & LOGIKA PREDIKSI
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODEL_TO_USE is None:
    st.error("⚠️ Sistem gagal dimuat atau model tidak ditemukan.")
    st.stop()

# --- BLOK CONTOH KATA (Dihapus dari Tampilan) ---
# Data keywords tetap didefinisikan di bagian 0
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
        
        prediction = MODEL_TO_USE.predict(X)[0]
        
        # 1. Styling Badge
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
            
        # 2. Highlight Teks Asli
        highlighted_text = highlight_keywords(input_text, SENTIMENT_KEYWORDS)
        
        # 3. Generate Penjelasan Panjang
        explanation_html = generate_long_explanation(prediction, input_text, CHOSEN_MODEL_NAME)


        # Tampilkan Kartu Hasil (Termasuk Penjelasan Panjang)
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <h4 style="color: #ccd6f6; margin-bottom: 5px;">HASIL ANALISIS SENTIMEN</h4>
                <div style='text-align: center;'>
                    <div class="sentiment-badge" style="background: {badge_bg};">
                        {icon} &nbsp; {label}
                    </div>
                </div>

                <div style="margin-top: 20px; padding: 10px; border: 1px dashed rgba(100, 255, 218, 0.2); border-radius: 8px;">
                    <p style="font-size: 14px; color: #64ffda; font-weight: 600;">📝 Teks Komentar yang Dianalisis:</p>
                    <p style="font-size: 15px; color: #ccd6f6; line-height: 1.6;">{highlighted_text}</p>
                </div>
                
                {explanation_html}
                
                <p style="text-align: right; margin-top: 15px;"><small style='color: #94a3b8;'>Model Klasifikasi: {CHOSEN_MODEL_NAME} (Optimasi GAM-GWO)</small></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
