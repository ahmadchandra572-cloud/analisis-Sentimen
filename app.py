st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #2980b9, #6dd5fa);
    }
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import joblib
import re
import string

# ============================
# Load Sastrawi Stemmer
# ============================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    SASTRAWI_AVAILABLE = True
except:
    stemmer = None
    SASTRAWI_AVAILABLE = False


# ============================
# Text Preprocessing
# ============================
@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URL, username, number, symbol
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Stemming
    if SASTRAWI_AVAILABLE and stemmer:
        text = stemmer.stem(text)

    return text.strip()


# ============================
# Load Vectorizer & Models
# ============================
@st.cache_resource
def load_files():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        models = {
            "Random Forest (RF)": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression (LR)": joblib.load("model_LR_GamGwo.pkl"),
            "Support Vector Machine (SVM)": joblib.load("model_SVM_GamGwo.pkl"),
        }

        return vectorizer, models
    except Exception as e:
        st.error(f"Gagal load model/vectorizer: {e}")
        return None, None


vectorizer, models = load_files()


# ============================
# UI
# ============================
st.title("Aplikasi Analisis Sentimen DPR")
st.write("Menggunakan model RF, LR, dan SVM (Optimasi GAM-GWO)")

if vectorizer is None or models is None:
    st.stop()

model_choice = st.selectbox("Pilih Model:", list(models.keys()))

text_input = st.text_area("Masukkan komentar:")

if st.button("Analisis Sentimen"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        cleaned = text_preprocessing(text_input)
        X = vectorizer.transform([cleaned])

        model = models[model_choice]
        prediction = model.predict(X)[0]

        st.info(f"Teks Bersih: {cleaned}")

        if prediction == "Positif":
            st.success(f"Sentimen: POSITIF 👍 ({model_choice})")
        elif prediction == "Negatif":
            st.error(f"Sentimen: NEGATIF 👎 ({model_choice})")
        else:
            st.warning(f"Sentimen: NETRAL 😐 ({model_choice})")
