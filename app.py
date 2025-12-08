import streamlit as st
import joblib
import re
import string

# ==========================================
# STEMMER (Sastrawi)
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
            "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
            "SVM": joblib.load("model_SVM_GamGwo.pkl")
        }
        return vectorizer, models
    except Exception as e:
        st.error(f"Gagal load model atau vectorizer: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# UI
# ==========================================
st.title("Sentiment Analyzer")
st.subheader("Text Sentiment Analysis App")

if VECTORIZER is None or MODELS is None:
    st.stop()

model_choice = st.selectbox("Choose Model:", list(MODELS.keys()))
input_text = st.text_area("Enter text:", height=120)

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.warning("Text cannot be empty!")
    else:
        cleaned_text = text_preprocessing(input_text)
        X = VECTORIZER.transform([cleaned_text])
        model = MODELS[model_choice]
        prediction = model.predict(X)[0]

        st.info(f"Cleaned Text: {cleaned_text}")

        if prediction.lower() == "positif":
            st.success("Sentiment: POSITIVE ✅")
        elif prediction.lower() == "negatif":
            st.error("Sentiment: NEGATIVE ❌")
        else:
            st.warning("Sentiment: NEUTRAL ⚪")
