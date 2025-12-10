import pandas as pd
import re

# =====================================
# 1. LOAD KAMUS KATA BAKU
# =====================================
def load_kamus():
    try:
        kamus = pd.read_excel("kamuskatabaku.xlsx")
        return dict(zip(kamus['tidak_baku'], kamus['kata_baku']))
    except Exception as e:
        print("Gagal load kamus:", e)
        return {}

# Load kamus langsung saat file dipakai
KAMUS = load_kamus()

# =====================================
# 2. CLEANING FUNCTIONS
# =====================================

def remove_emoji(text):
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U0001F004-\U0001F0CF"
        u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text) if isinstance(text, str) else ""

def remove_numbers(text):
    return re.sub(r'\d+', '', text) if isinstance(text, str) else ""

def remove_username(text):
    return re.sub(r'@[^\s]+', '', text) if isinstance(text, str) else ""

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else ""

def remove_html(text):
    return re.sub(r'<.*?>', '', text) if isinstance(text, str) else ""

# =====================================
# 3. NORMALISASI KATA TIDAK BAKU
# =====================================
def normalisasi_kata(text):
    if not isinstance(text, str):
        return ""

    words = text.split()
    hasil = []

    for word in words:
        if word in KAMUS:
            hasil.append(KAMUS[word])
        else:
            hasil.append(word)

    return ' '.join(hasil)

# =====================================
# 4. PIPELINE PREPROCESSING UNTUK MODEL
# =====================================
def full_preprocessing(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_emoji(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_username(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = normalisasi_kata(text)

    return text.strip()
