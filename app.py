import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk
import plotly.express as px
from streamlit_option_menu import option_menu
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Analisis Emosi MBG", page_icon="📊", layout="wide")

# --- 2. LOAD MODEL & RESOURCE (CACHE) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
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

# Inisialisasi Resource
download_nltk_data()
tfidf, svm_model = load_models()
stemmer = load_stemmer()

# Setup Stopwords dengan Pengecualian (Sesuai Arahan Dospem)
list_stopwords = set(stopwords.words('indonesian'))
kata_penting = {'tidak', 'bukan', 'belum', 'kurang', 'sangat', 'tepat', 'jangan'}
list_stopwords = list_stopwords - kata_penting

# --- 3. FUNGSI PREPROCESSING ---
def preprocess_input(text):
    text = str(text).lower()
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

# --- 4. MEMBUAT MENU DI SIDEBAR ---
with st.sidebar:
    # Membuat menu modern dengan streamlit-option-menu
    pilihan_menu = option_menu(
        menu_title="Navigasi Sistem",  
        options=["Dashboard Emosi", "Analisis Tunggal", "Analisis Batch", "Tentang Sistem"], 
        icons=["bar-chart-fill", "chat-text-fill", "file-earmark-spreadsheet-fill", "info-circle-fill"], 
        menu_icon="compass", 
        default_index=0, 
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "#ff4b4b", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
        }
    )
    
    st.divider()
    st.caption("Dikembangkan oleh:")
    st.write("**Reinardus Galentio Axelle**")
    st.write("Informatika - Univ. Gunadarma (2026)")

# --- 5. KONTEN HALAMAN BERDASARKAN MENU ---

if pilihan_menu == "Dashboard Emosi":
    st.title("📈 Dashboard Analisis Emosi Program MBG")
    st.markdown("Ringkasan distribusi emosi warganet pada media sosial X terhadap kebijakan **Makan Bergizi Gratis (MBG)** berdasarkan data historis pelabelan EmoLex.")
    
    # Data Historis (Berdasarkan PDF Revisi 3)
    data_emosi = {
        'Emosi': ['Trust', 'Anticipation', 'Anger', 'Joy', 'Sadness', 'Fear', 'Disgust', 'Surprise'],
        'Jumlah': [1735, 1469, 846, 715, 614, 611, 604, 155]
    }
    df_chart = pd.DataFrame(data_emosi)
    
    # Tampilkan Angka Metrik di atas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Berlabel", "6.749 Tweet")
    col2.metric("Emosi Tertinggi", "Trust (26%)")
    col3.metric("Akurasi Model", "93.80%")
    col4.metric("Algoritma", "SVM (Linear)")
    
    st.divider()
    
    # Tampilkan Grafik Bar Chart dan Pie Chart berdampingan
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_bar = px.bar(df_chart, x='Emosi', y='Jumlah', color='Emosi', title="Grafik Distribusi Kategori Emosi (Bar Chart)")
        st.plotly_chart(fig_bar, width="stretch")
        
    with col_chart2:
        fig_pie = px.pie(df_chart, names='Emosi', values='Jumlah', title="Persentase Distribusi Emosi (Pie Chart)", hole=0.3)
        st.plotly_chart(fig_pie, width="stretch")

elif pilihan_menu == "Analisis Tunggal":
    st.title("📝 Klasifikasi Emosi Warganet (Tunggal)")
    st.markdown("Masukkan teks cuitan (*tweet*) atau kalimat opini untuk mengetahui emosi dominan yang terkandung di dalamnya secara instan.")
    
    teks_input = st.text_area("Masukkan Teks di bawah ini:", height=150, placeholder="Ketik kalimat di sini...")
    
    if st.button("Analisis Emosi", type="primary", width="stretch"):
        if teks_input.strip():
            with st.spinner('Sedang memproses dan menganalisis teks...'):
                teks_bersih = preprocess_input(teks_input)
                vektor_input = tfidf.transform([teks_bersih])
                hasil_emosi = svm_model.predict(vektor_input)[0]
            
            st.divider()
            st.subheader("Hasil Analisis:")
            st.info(f"**Teks Bersih (Hasil Preprocessing):**\n\n*{teks_bersih}*")
            st.success(f"**Emosi Dominan Terdeteksi:** {hasil_emosi.upper()}")
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu sebelum menekan tombol analisis.")

elif pilihan_menu == "Analisis Batch":
    st.title("📂 Analisis Data Massal (Batch Processing)")
    st.markdown("Unggah file berformat CSV yang berisi banyak data teks (misalnya data hasil *crawling* mentah) untuk diklasifikasikan emosinya sekaligus.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file CSV. Pastikan format file benar. Error detail: {e}")
            st.stop()
            
        st.write("✅ File berhasil diunggah. Pratinjau 5 baris pertama:")
        st.dataframe(df_upload.head())
        
        # Pilih kolom yang berisi teks
        kolom_teks = st.selectbox("Pilih nama kolom yang berisi teks cuitan/opini:", df_upload.columns)
        
        if st.button("Mulai Proses Klasifikasi Batch", type="primary"):
            try:
                with st.spinner('Memproses ribuan baris teks (ini mungkin memakan waktu beberapa menit)...'):
                    df_upload['Teks_Bersih'] = df_upload[kolom_teks].apply(preprocess_input)
                    
                    vektor_batch = tfidf.transform(df_upload['Teks_Bersih'])
                    prediksi_batch = svm_model.predict(vektor_batch)
                    
                    df_upload['Prediksi_Emosi'] = prediksi_batch
                    
                st.success(f"🎉 Selesai! {len(df_upload)} baris data berhasil diklasifikasikan.")
                st.dataframe(df_upload[[kolom_teks, 'Prediksi_Emosi']].head(10))
                
                csv_hasil = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Unduh Hasil Klasifikasi (CSV)",
                    data=csv_hasil,
                    file_name='hasil_klasifikasi_mbg.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses Machine Learning: {e}")

elif pilihan_menu == "Tentang Sistem":
    st.title("ℹ️ Metodologi & Informasi Sistem")
    st.write("Aplikasi ini merupakan implementasi dari model *Machine Learning* yang dikembangkan untuk penelitian skripsi.")
    
    st.subheader("1. Detail Algoritma")
    st.write("- **Algoritma Klasifikasi:** Support Vector Machine (SVM) dengan Kernel Linear.")
    st.write("- **Ekstraksi Fitur:** Term Frequency-Inverse Document Frequency (TF-IDF).")
    st.write("- **Penanganan Data Imbalance:** Synthetic Minority Over-sampling Technique (SMOTE).")
    st.write("- **Pendekatan Leksikon:** NRC Emotion Lexicon (EmoLex) untuk pelabelan 8 kategori emosi (Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust).")
    
    st.subheader("2. Performa Model")
    st.write("Berdasarkan pengujian pada data uji sebanyak **2.776 sampel**, model menghasilkan metrik evaluasi sebagai berikut:")
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Akurasi (Accuracy)", "93.80%")
    col_met2.metric("Rata-rata Presisi", "94.00%")
    col_met3.metric("Rata-rata F1-Score", "94.00%")
    
    st.info("Kinerja terbaik (F1-Score: 0.99) diperoleh pada pendeteksian emosi **Surprise**, berkat efektivitas penyeimbangan data menggunakan metode SMOTE.")