import pandas as pd
import re

# =============================
# LOAD KAMUS
# =============================
def load_kamus():
    try:
        kamus = pd.read_excel("kamuskatabaku.xlsx")
        return dict(zip(kamus['tidak_baku'], kamus['kata_baku']))
    except:
        return {}

# =============================
# CLEANING FUNCTIONS
# =============================
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
