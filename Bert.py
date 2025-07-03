
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Fungsi load model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    # Load tokenizer dari folder
    tokenizer = BertTokenizer.from_pretrained("Tugas_Akhir/bert_tokenizer")

    # Load model dan bobotnya
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load("Tugas_Akhir/bert_review_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

# Load resources
tokenizer, model = load_model_tokenizer()

# UI Streamlit
st.title("🧠 BERT Review Classifier")
st.write("Masukkan teks review, dan model BERT akan mengklasifikasikannya ke: **Original** atau **Fake**.")

# Input pengguna
user_input = st.text_area("✍️ Masukkan Review Teks di sini:", height=150)

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        # Tokenisasi
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        # Label mapping
        label_map = {0: "Original", 1: "Fake"}
        st.success(f"🧾 Prediksi: **{label_map[pred_class]}**")
        st.write(f"📊 Confidence Score: `{probs[0][pred_class]:.4f}`")

