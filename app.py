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

# --- FUNGSI UTILITY ---
def update_input_from_selectbox_asli():
    selected_value = st.session_state.selected_sample_asli
    if selected_value != "-- PILIH DATASET ASLI --":
        st.session_state.current_input = selected_value
        st.session_state.selected_sample_baku = "-- PILIH DATASET BAKU --"

def update_input_from_selectbox_baku():
    selected_value = st.session_state.selected_sample_baku
    if selected_value != "-- PILIH DATASET BAKU --":
        st.session_state.current_input = selected_value
        st.session_state.selected_sample_asli = "-- PILIH DATASET ASLI --"

# --- INISIALISASI SESSION STATE ---
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""
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
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #8892b0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

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

.model-label-box {
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid rgba(100, 255, 218, 0.2);
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

# Pilih Random Forest sebagai model utama
CHOSEN_MODEL_NAME = "Random Forest"
MODEL_TO_USE = MODELS[CHOSEN_MODEL_NAME] if MODELS and CHOSEN_MODEL_NAME in MODELS else None


# ==========================================
# 3.5️⃣ DAFTAR DATASET (ASLI & BAKU)
# ==========================================
SAMPLE_COMMENTS_ASLI = [
    'Dpr jancok dpr tidak adil dasar', 'Setuju gaji anggota dewan umr supaya orang tidak ambisisius',
    'Brukakaka 1000% bayar PBB. Yang tinggal di kolong jembatan layak gk.', 'Mantap tarian jogetnya. Macam monyet dapat pisang.',
    'Apa dpr . Mau jaga rakyat . Atau mau siksa rakyat .', 'Kadang memang bikin hati panas dan ngerasa nggak adil.',
    'Semoga dapet musibah gaji dpr naik udh enak ksh tunjangan hidup', 'Pejabat paling terkorup indonesia.',
    'Hebat ...anggota DPR bisa makmur.. dan suksesss', 'Kalau perkara rujab gk layak huni,, kan bisa uang kompensasi nya dijadiin buat renovasi',
    'Saya lebih setuju gajih TNI di naikan TNI nyawa taruhannya', 'Mana dulu yang bilang klok probowo jadi presiden indo bakal makmur',
    'BUBARKAN MRP DPR. ALOKASI GAJI MERRKA BUAT RAKYAT', 'Masih banyak pengangguran, masih ada yang kelaparan ,hebat',
    'Mantap lanjutkan', 'Apa sih fungsi DPR MPR?', 'SI EKO PATRIO DAN GEROMBOLANYA ANGGOTA DPR PALING KOPLAK',
    'Ga adil klu cuman anggota DPR yg naik guru tuh harusnya yg dinaikan', 'Gajih tidak naik aja orang masih banyak yang ingin jadi DPR apa lagi naik??',
    'Rakyat kaya DPR miskin', 'Joget2 gaji naik, sangat menyakitkan rakyat miskin', 'yg kaya makin kaya yg miskin makin miskin',
    'tenyata ge baru tau arti APK (Anjayyy Pemerintah Korupsi)', 'CUIHHH NAJIS JOGED2 DI ATAS PENDERITAAN RAKYAT',
    'Pejabat makin kaya, Masyarakatnya mkin pada miskin', 'itulah anggota dpr pintar bersilat lidah, gaji di ganti nama menjadi tunjangan',
    'Bikin miskin rakyat aja n dewan...', 'Yang kaya makin kaya..yang miskin makin miskin', 'kenaikan tunjangan anggota dpr dibayar dengan menaikan pajak',
    'Bubar kn DPR penipu ranya ngk ada gunanya', 'Negara konoha benar benar negara bobrok', 'DPR enak pa naik gaji saya warga miskin kesusahan',
    'Puan bau tanah', 'Pantasan Rakyat pada marah kaya gini 😭😭😭'
]

SAMPLE_COMMENTS_BAKU = [
    'Kebijakan kenaikan tunjangan anggota DPR harus mempertimbangkan kondisi ekonomi masyarakat.',
    'Transparansi anggaran dalam pengalokasian dana fasilitas perumahan anggota dewan sangat diperlukan.',
    'Seharusnya pemerintah lebih mengutamakan peningkatan kesejahteraan guru honorer.',
    'Masyarakat menaruh harapan besar agar anggota DPR menolak fasilitas mewah.',
    'Integritas wakil rakyat diuji melalui keberanian mereka dalam membatasi pengeluaran anggaran.',
    'Standardisasi gaji pejabat publik harus mengacu pada kemampuan fiskal negara.',
    'Rakyat menghendaki adanya pengawasan ketat terhadap setiap aliran dana tunjangan anggota legislatif.',
    'Kepuasan publik terhadap kinerja parlemen merupakan indikator keberhasilan demokrasi di Indonesia.',
    'Pejabat negara harus memiliki rasa tanggung jawab yang besar dalam mengelola harta kekayaan negara.',
    'Pengabdian tulus tanpa mengharapkan kemewahan adalah ciri negarawan sejati yang dirindukan rakyat.',
    'Kesejahteraan masyarakat pedesaan harus menjadi target utama dalam setiap kebijakan fiskal nasional.',
    'Setiap anggota dewan harus mempertanggungjawabkan setiap dana yang diterima kepada konstituennya.',
    'Kebijakan yang memihak pada kepentingan elit politik akan mencederai semangat reformasi bangsa.',
    'Efisiensi belanja pegawai merupakan langkah strategis untuk mengurangi defisit anggaran pendapatan negara.',
    'Transformasi digital di parlemen seharusnya mampu mengurangi biaya operasional yang tidak perlu.',
    'Keadilan bagi tenaga honorer harus diperjuangkan sejajar dengan pembahasan tunjangan pejabat.',
    'Institusi DPR harus menjadi simbol kesederhanaan and kerja keras bagi seluruh rakyat Indonesia.',
    'Masa depan bangsa ditentukan oleh keberanian pemimpin dalam mengambil keputusan yang adil and jujur.',
    'Pertumbuhan ekonomi yang inklusif hanya bisa dicapai melalui tata kelola keuangan yang transparan.',
    'Rakyat akan selalu mendukung kebijakan pemerintah yang benar-benar berfokus pada kebutuhan dasar warga.',
    'Kritik terhadap kenaikan tunjangan adalah bagian dari hak demokrasi yang dijamin oleh undang-undang.',
    'Integritas and dedikasi harus menjadi landasan utama bagi setiap penyelenggara negara di Indonesia.',
    'Pemerintah perlu meninjau kembali urgensi pemberian fasilitas tambahan bagi pejabat di masa krisis.',
    'Kesadaran kolektif untuk melakukan penghematan anggaran harus dimulai dari jajaran pimpinan tertinggi.',
    'Distribusi pendapatan yang merata akan menjamin stabilitas keamanan and ketertiban masyarakat.',
    'Program jaminan sosial bagi rakyat miskin lebih mendesak untuk didanai daripada tunjangan mewah.',
    'Keputusan politik yang bijak adalah yang mampu menyentuh hati rakyat and memberikan solusi nyata.',
    'Setiap undang-undang yang disahkan harus memberikan dampak positif bagi kemajuan ekonomi rakyat.',
    'Transparansi publik dalam hal gaji pejabat akan mengurangi potensi penyalahgunaan wewenang.',
    'Sikap empati terhadap kesulitan warga adalah modal utama bagi seorang wakil rakyat yang amanah.',
    'Keberlanjahan fiskal negara sangat bergantung pada ketepatan dalam menentukan prioritas belanja.',
    'Rakyat menuntut adanya efektivitas kerja yang nyata dari seluruh anggota dewan yang terhormat.',
    'Peningkatan taraf hidup masyarakat bawah harus menjadi misi utama setiap wakil rakyat di parlemen.',
    'Ketimpangan sosial adalah musuh demokrasi yang harus dilawan dengan kebijakan anggaran yang adil.',
    'Negara harus hadir dalam memberikan perlindungan ekonomi bagi setiap warga negara tanpa kecuali.',
    'DPR diharapkan menjadi jembatan aspirasi yang kokoh antara rakyat and pemerintah pusat.',
    'Etika politik melarang penggunaan anggaran negara untuk kepentingan kemewahan pribadi pejabat.',
    'Penyederhanaan birokrasi and anggaran adalah kunci efisiensi dalam pemerintahan modern.',
    'Rakyat akan menghormati pemimpin yang berani hidup sederhana di tengah kesulitan bangsanya.',
    'Kebijakan anggaran yang pro-rakyat akan meningkatkan martabat bangsa di mata internasional.',
    'Setiap anggota legislatif wajib menjunjung tinggi nilai-nilai kejujuran dalam mengelola dana negara.',
    'Keberhasilan pembangunan nasional diukur dari berkurangnya angka kemiskinan and pengangguran.',
    'DPR harus menjadi contoh dalam implementasi tata kelola lembaga yang bersih and akuntabel.',
    'Masyarakat berharap adanya reformasi dalam sistem pemberian tunjangan bagi seluruh pejabat negara.',
    'Kekuatan sebuah negara terletak pada kepercayaan rakyatnya terhadap para pemimpin di parlemen.',
    'Sikap kritis masyarakat adalah pendorong utama bagi perbaikan kinerja institusi pemerintahan.',
    'Penghapusan fasilitas yang berlebihan merupakan langkah nyata dalam mendukung penghematan nasional.',
    'Pembangunan sumber daya manusia lebih penting daripada pemenuhan fasilitas mewah elit politik.',
    'Pemerintah and DPR harus bekerja sama dalam menciptakan kebijakan fiskal yang sehat and kuat.',
    'Amanah rakyat harus dijalankan dengan penuh rasa syukur and dedikasi tinggi bagi nusa and bangsa.',
    'Setiap kebijakan ekonomi harus diarahkan pada terciptanya lapangan kerja baru bagi generasi muda.',
    'Kepedulian terhadap lingkungan and masalah sosial harus tecermin dalam anggaran pembangunan.',
    'DPR wajib memastikan bahwa pajak yang dibayarkan rakyat kembali dalam bentuk pelayanan publik berkualitas.',
    'Moralitas and etika harus selalu mendahului kepentingan politik dalam setiap rapat paripurna.',
    'Kepercayaan publik adalah aset yang sangat mahal and harus dijaga dengan kinerja yang nyata.',
    'Sistem demokrasi akan berjalan baik jika ada transparansi penuh dalam penggunaan dana negara.',
    'Kenaikan tunjangan tidak boleh menjadi alasan bagi penurunan semangat pengabdian kepada rakyat.',
    'Rakyat merindukan sosok pemimpin yang lebih banyak bekerja daripada menuntut fasilitas tambahan.',
    'Indonesia yang maju hanya bisa terwujud jika seluruh elemen bangsa memiliki semangat keadilan sosial.',
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
        input_text = st.text_area("Ketik Komentar", value=st.session_state.current_input, placeholder="Ketik di sini...", height=100, key="current_input_area", label_visibility="collapsed")
        analyze_button = st.button("🔍 ANALISIS SEKARANG")

if analyze_button:
    current_val = st.session_state.current_input_area if st.session_state.current_input_area else st.session_state.current_input
    if current_val.strip() == "":
        st.warning("⚠️ Masukkan komentar!")
    else:
        # Proses Prediksi
        clean_text = text_preprocessing(current_val)
        X = VECTORIZER.transform([clean_text])
        
        # LABEL ASLI DARI MODEL
        ml_prediction = MODEL_TO_USE.predict(X)[0]
        
        # LABEL AKHIR SETELAH KOREKSI
        final_prediction = force_correct_prediction(clean_text, ml_prediction)
        
        if 'koreksi' in final_prediction.lower() or final_prediction.lower() == "negatif":
            label, badge_bg, icon = "NEGATIF", "linear-gradient(90deg, #dc2626, #f87171)", "😡"
        elif final_prediction.lower() == "positif":
            label, badge_bg, icon = "POSITIF", "linear-gradient(90deg, #059669, #34d399)", "😊"
        else:
            label, badge_bg, icon = "NETRAL", "linear-gradient(90deg, #64748b, #94a3b8)", "😐"

        # Tampilkan Hasil Gabungan (Label Akhir + Label Asli Model)
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <h4 style="color: #ccd6f6; margin-bottom: 5px;">HASIL ANALISIS SENTIMEN</h4>
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
                <div class="model-label-box">
                    <p style="color: #8892b0; font-size: 14px; margin-bottom: 0;">
                        Label Asli (Model): <span style="color: #64ffda; font-weight: 600;">{ml_prediction.upper()}</span>
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
