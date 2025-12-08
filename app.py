import streamlit as st
import joblib
import re
import string
import base64

# ===============================
# BACKGROUND IMAGE
# ===============================
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
        background-color: #2c7be5;
        color: white;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# STEMMER (SASTRAWI)
# ===============================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    STEMMER = StemmerFactory().create_stemmer()
except:
    STEMMER = None

# ===============================
# PREPROCESSING
# ===============================
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    if STEMMER:
        text = STEMMER.stem(text)

    return text.strip()

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    models = {
        "Random Forest (RF)": joblib.load("model_RF_GamGwo.pkl"),
        "Logistic Regression (LR)": joblib.load("model_LR_GamGwo.pkl"),
        "Support Vector Machine (SVM)": joblib.load("model_SVM_GamGwo.pkl"),
    }
    return vectorizer, models

vectorizer, models = load_models()

# ===============================
# UI
# ===============================
st.title("Sentiment Analyzer")
st.subheader("Using RF, LR & SVM Models")

model_choice = st.selectbox("Choose Model:", list(models.keys()))
input_text = st.text_area("Enter text:")

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.warning("Text cannot be empty!")
    else:
        clean_text = text_preprocessing(input_text)
        X = vectorizer.transform([clean_text])
        prediction = models[model_choice].predict(X)[0]

        st.info(f"Processed Text: {clean_text}")

        if prediction.lower() == "positif":
            st.success("Result: POSITIVE ✅")
        elif prediction.lower() == "negatif":
            st.error("Result: NEGATIVE ❌")
        else:
            st.warning("Result: NEUTRAL ⚖️")
