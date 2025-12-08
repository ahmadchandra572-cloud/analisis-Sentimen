import streamlit as st
import joblib
import re
import string
import base64

# ==========================================
# LOAD BACKGROUND IMAGE
# ==========================================
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64_of_bin_file("gamabr.jpg")

# ==========================================
# BACKGROUND STYLE + SIDE IMAGE
# ==========================================
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg, #1f2937, #111827);
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 200%;
    height: 200%;
    background-image: repeating-linear-gradient(
        45deg,
        rgba(255,255,255,0.03),
        rgba(255,255,255,0.03) 2px,
        transparent 2px,
        transparent 40px
    );
    z-index: -1;
}}

.block-container {{
    background-color: rgba(17, 24, 39, 0.85);
    padding: 2rem;
    border-radius: 16px;
    max-width: 800px;
}}

.side-image {{
    position: fixed;
    right: 20px;
    bottom: 20px;
    width: 200px;
    opacity: 0.15;
    z-index: -1;
}}
</style>

<img class="side-image" src="data:image/jpg;base64,{bg_img}">

""", unsafe_allow_html=True)

# ==========================================
# STEMMER
# ==========================================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    STEMMER = StemmerFactory().create_stemmer()
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
# LOAD MODELS
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
