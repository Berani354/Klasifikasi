import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = joblib.load('Tugas_Akhir/logistic_model.pkl')
vectorizer = joblib.load('Tugas_Akhir/tfidf_vectorizer.pkl')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # hapus tag HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus angka dan simbol
    text = re.sub(r'\s+', ' ', text)  # hilangkan spasi ganda
    return text.strip()

# Fungsi prediksi
def predict_review(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "FAKE" if prediction == 1 else "REAL"

# UI Streamlit
st.set_page_config(page_title="Deteksi Ulasan Palsu", layout="centered")
st.title("🕵️‍♂️ Deteksi Ulasan Palsu Produk Online")

review_input = st.text_area("Masukkan teks ulasan produk:", height=200)

if st.button("Deteksi"):
    if review_input.strip():
        result = predict_review(review_input)
        if result == "FAKE":
            st.error("⚠️ Ulasan ini terdeteksi PALSU")
        else:
            st.success("✅ Ulasan ini terdeteksi ASLI")
    else:
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
