import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Emosi MBG", page_icon="📊")

# --- 1. LOAD MODEL & RESOURCE (Di-cache agar super cepat) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    svm_model = joblib.load('model_svm_emosi.pkl')
    return tfidf, svm_model

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

# Inisialisasi semua resource
download_nltk_data()
tfidf, svm_model = load_models()
stemmer = load_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

# --- 2. FUNGSI PREPROCESSING ---
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens_no_stop = [word for word in tokens if word not in list_stopwords]
    
    text_digabung = ' '.join(tokens_no_stop)
    text_bersih = stemmer.stem(text_digabung)
    return text_bersih

# --- 3. ANTARMUKA WEB STREAMLIT ---
st.title("📊 Klasifikasi Emosi Warganet")
st.markdown("**Topik:** Program Makan Bergizi Gratis (MBG) | **Algoritma:** Support Vector Machine (Linear)")
st.divider()

# Kotak Input Teks
teks_input = st.text_area("Masukkan Teks Tweet / Sentimen di bawah ini:", height=150, placeholder="Ketik kalimat di sini...")

# Tombol Prediksi
if st.button("Analisis Emosi", type="primary", use_container_width=True):
    if teks_input.strip():
        with st.spinner('Sedang memproses dan menganalisis teks...'):
            # Pembersihan teks
            teks_bersih = preprocess_input(teks_input)
            
            # Prediksi dengan model SVM
            vektor_input = tfidf.transform([teks_bersih])
            hasil_emosi = svm_model.predict(vektor_input)[0]
        
        # Tampilkan Hasil
        st.divider()
        st.subheader("Hasil Analisis:")
        
        # Menampilkan teks yang sudah dibersihkan
        st.info(f"**Teks Bersih (Hasil Preprocessing):**\n\n*{teks_bersih}*")
        
        # Menampilkan label emosi dengan warna yang menonjol
        st.success(f"**Emosi Dominan Terdeteksi:** {hasil_emosi.upper()}")
        
    else:
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu sebelum menekan tombol analisis.")