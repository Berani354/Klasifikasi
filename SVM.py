import streamlit as st
import joblib
import numpy as np

# Load model dan vectorizer
svm_model = joblib.load("Tugas_Akhir/SVM/svm_model.pkl")
vectorizer = joblib.load("Tugas_Akhir/SVM/tfidf_vectorizer.pkl")

# Judul aplikasi
st.title("Deteksi Ulasan Palsu")
st.subheader("Menggunakan Model SVM dan TF-IDF")

# Input teks ulasan
review_text = st.text_area("Masukkan teks ulasan produk di sini:")

# Tombol prediksi
if st.button("Prediksi"):
    if review_text.strip() == "":
        st.warning("Harap masukkan teks terlebih dahulu.")
    else:
        # Preprocessing dan prediksi
        vectorized_text = vectorizer.transform([review_text])
        prediction = svm_model.predict(vectorized_text)

        # Tampilkan hasil
        label = "🟢 Ulasan Asli (OR)" if prediction[0] == 0 else "🔴 Ulasan Palsu (CG)"
        st.markdown(f"### Hasil Prediksi: {label}")
