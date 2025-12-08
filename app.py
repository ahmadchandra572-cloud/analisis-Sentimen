import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# BACKGROUND IMAGE SETUP
# ==========================================
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_of_bin_file("gamabr.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    h1, h2, h3, p, label {{
        color: white !important;
    }}

    .stTextArea textarea {{
        background-color: rgba(0,0,0,0.6);
        color: white;
    }}

    .stSelectbox div {{
        background-color: rgba(0,0,0,0.6);
        color: white;
    }}

    .stButton button {{
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SASTRAWI (STEMMING)
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

# ==========================================
# PREPROCESSING
# ==========================================
@st.cache_data
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
# LOAD MODEL & VECTORIZER
# ==========================================
@st.cache_resource
def load_resources():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        models = {
            "Random Forest (RF)": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression (LR)": joblib.load("model_LR_GamGwo.pkl"),
            "Support Vector Machine (SVM)": joblib.load("model_SVM_GamGwo.pkl"),
        }
        return vectorizer, models
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# UI APLIKASI
# ==========================================
st.title("Sentiment Analyzer")
st.subheader("Model: RF, LR & SVM (GAM-GWO Optimized)")

if VECTORIZER is None or MODELS is None:
    st.stop()

model_choice = st.selectbox("Pilih Model:", list(MODELS.keys()))
input_text = st.text_area("Masukkan Teks:")

if st.button("Analisis Sentimen"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        clean_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([clean_text])
        prediction = MODELS[model_choice].predict(X)[0]

        st.info(f"Teks Bersih: {clean_text}")

        if prediction.lower() == "positif":
            st.success("Sentimen: POSITIVE ✅")
        elif prediction.lower() == "negatif":
            st.error("Sentimen: NEGATIVE ❌")
        else:
            st.warning("Sentimen: NEUTRAL ⚖️")
