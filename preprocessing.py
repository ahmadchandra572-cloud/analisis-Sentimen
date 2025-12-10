import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ======================
# LOAD STEMMER
# ======================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ======================
# CLEANING FUNCTIONS
# ======================
def remove_emoji(text):
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_emoji(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text.strip()

# ======================
# NORMALISASI
# ======================
def load_kamus():
    kamus = pd.read_excel("kamuskatabaku.xlsx")
    return dict(zip(kamus['tidak_baku'], kamus['kata_baku']))

def normalize_text(text, kamus):
    words = text.split()
    return " ".join([kamus.get(w, w) for w in words])

# ======================
# TOKEN + STEM
# ======================
def stem_text(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

# ======================
# MAIN PIPELINE
# ======================
def full_preprocess(text, kamus):
    text = clean_text(text)
    text = normalize_text(text, kamus)
    text = stem_text(text)
    return text
