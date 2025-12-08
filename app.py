import streamlit as st
import joblib
import re
import string
import base64 # <-- TAMBAHAN: Untuk meng-encode gambar

# ==========================================
# 0️⃣ BACKGROUND IMAGE ENCODER
# ==========================================
def get_base64_of_bin_file(file_path):
    """Mengubah file gambar menjadi string Base64."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None # Mengembalikan None jika file tidak ditemukan

# Asumsi nama file adalah 'gamabr' dan formatnya jpg. Harap dicek di repo!
BG_IMAGE_FILENAME = "gamabr" 
BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME) 


# ==========================================
# BACKGROUND STYLE (MODIFIED TO USE IMAGE)
# ==========================================
if BG_IMAGE_B64:
    # Jika gambar berhasil dimuat, gunakan gambar sebagai background
    BG_STYLE = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(17, 24, 39, 0.85); /* Agar teks di konten tetap terbaca */
        padding: 2rem;
        border-radius: 16px;
    }}
    h1, h2, h3, label, p {{
        color: #e5e7eb !important;
    }}
    </style>
    """
else:
    # Jika gambar gagal dimuat, kembali ke background gradient default Anda
    BG_STYLE = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f2937, #111827);
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(17, 24, 39, 0.85);
        padding: 2rem;
        border-radius: 16px;
    }
    h1, h2, h3, label, p {
        color: #e5e7eb !important;
    }
    </style>
    """

st.markdown(BG_STYLE, unsafe_allow_html=True)
# ==========================================
# STEMMER
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================
@st.cache_data
# ... (lanjutkan fungsi text_preprocessing) ...
def text_preprocessing(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode('ascii')

    if STEMMER:
        text = STEMMER.stem(text)

    return text.strip()

# ==========================================
# LOAD MODEL DAN VECTORIZER
# ==========================================
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
        st.error(f"Gagal load file: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# UI
# ==========================================
st.title("Sentiment Analyzer")
st.subheader("Text Sentiment Analysis App")

if VECTORIZER is None or MODELS is None:
    st.stop()

model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
input_text = st.text_area("Enter text here", height=120)

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.warning("Text cannot be empty!")
    else:
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]

        st.info(f"Cleaned Text: {clean_text}")

        if prediction.lower() == "positif":
            st.success("Sentiment: POSITIVE ✅")
        elif prediction.lower() == "negatif":
            st.error("Sentiment: NEGATIVE ❌")
        else:
            st.warning("Sentiment: NEUTRAL ⚪")
