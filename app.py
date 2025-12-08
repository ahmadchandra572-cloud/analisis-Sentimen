import streamlit as st
import joblib
import re
import string
# PENTING: Sastrawi harus diimpor untuk Stemming.
# Jika ini error (ModuleNotFoundError), Anda wajib menambahkan 'Sastrawi' ke requirements.txt.
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
    # Inisialisasi Stemmer di luar fungsi untuk efisiensi
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except ImportError:
    st.warning("Peringatan: Library Sastrawi tidak ditemukan. Proses Stemming akan dilewati.")
    STEMMER = None
except Exception as e:
    st.warning(f"Peringatan: Gagal inisialisasi Sastrawi. Stemming dilewati. Error: {e}")
    STEMMER = None


# ==========================================
# 0️⃣ PREPROCESSING FUNCTION (Wajib Sama dengan Training)
# ==========================================
@st.cache_data
def text_preprocessing(text):
    """Membersihkan dan melakukan stemming pada teks input."""
    if not isinstance(text, str): 
        return ""
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Cleaning (Hapus URL, username, angka, dan simbol)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+','', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 3. Stemming (Sastrawi)
    if STEMMER:
        text = STEMMER.stem(text)
        
    return text.strip()

# ==========================================
# 1️⃣ LOAD SEMUA MODEL DAN VECTORIZER
# ==========================================
# Menggunakan st.cache_resource agar loading file .pkl hanya 1x
@st.cache_resource
def load_resources():
    try:
        # Load Vectorizer ke variabel yang benar (PENTING!)
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # Load 3 MODEL OPTIMASI (Menggunakan NAMA FILE YANG BENAR di repo)
        models = {
            "Random Forest (RF)": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression (LR)": joblib.load("model_LR_GamGwo.pkl"),
            "Support Vector Machine (SVM)": joblib.load("model_SVM_GamGwo.pkl"),
        }
        # Mengembalikan Vectorizer dan Dictionary Models
        return vectorizer, models
    except Exception as e:
        # Menampilkan error yang jelas jika ada masalah loading file
        st.error(f"FATAL ERROR: Gagal memuat file .pkl. Pastikan semua file .pkl diupload dan nama file sudah benar. Detail: {e}")
        return None, None
            
# Panggil fungsi load dan simpan hasilnya di variabel global
VECTORIZER, MODELS = load_resources()


# ==========================================
# 2️⃣ ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("Aplikasi Analisis Sentimen DPR")
st.subheader("Model Optimasi GAM-GWO")

# Cek apakah resource berhasil dimuat
if MODELS is None or VECTORIZER is None:
    st.warning("Aplikasi tidak dapat berjalan karena error pemuatan file. Harap cek log Streamlit.")
    st.stop() # Hentikan eksekusi script lebih lanjut

# Pilihan Model menggunakan Selectbox
model_options = list(MODELS.keys())
model_choice = st.selectbox("Pilih Algoritma Prediksi:", model_options)

input_text = st.text_area("Masukkan Komentar YouTube di sini:", height=100)

if st.button("analisis-Sentimen"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
        
    else:
        # --- PREDICTIVE LOGIC ---
        
        # 1. Preprocessing Input
        clean_text = text_preprocessing(input_text)
        
        # 2. Vectorization (TF-IDF digunakan di sini, disembunyikan dari user)
        X = VECTORIZER.transform([clean_text])
        
        # 3. Prediksi
        selected_model = MODELS[model_choice]
        prediction = selected_model.predict(X)[0]

        # 4. Tampilkan Hasil
        st.info(f"Teks Bersih (Preprocessed): {clean_text}")

        # Streamlit akan selalu menampilkan "Netral" jika model gagal membedakan sentimen
        if prediction == "Positif":
            st.success(f"Sentimen: POSITIF 👍 (Model: {model_choice})")
        elif prediction == "Negatif":
            st.error(f"Sentimen: NEGATIF 👎 (Model: {model_choice})")
        else:
            st.warning(f"Sentimen: NETRAL 😐 (Model: {model_choice})")
