import streamlit as st
import pickle
import numpy as np
import re
import time
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

# ========== CUSTOM LAYER UNTUK MENGATASI QUANTIZATION_CONFIG ==========
class QuantizationAwareEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        # Hapus parameter 'quantization_config' jika ada (penyebab error)
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.pop('quantization_config', None)
        return config

# ========== FUNGSI CLEANING TEKS ==========
def clean_text(text, stemmer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+|[^\w\s]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# ========== LOAD RESOURCES (CACHED) ==========
@st.cache_resource
def load_resources():
    try:
        # Download stopwords (hanya sekali)
        nltk.download('stopwords', quiet=True)
        
        # Stemmer dan stopwords
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stop_words = set(stopwords.words('indonesian'))
        
        # Tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Model LSTM dengan custom_objects
        model = tf.keras.models.load_model(
            'lstm_model_whatsapp.keras',
            custom_objects={'Embedding': QuantizationAwareEmbedding},
            compile=False  # compile ulang sendiri
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return {'stemmer': stemmer, 'stop_words': stop_words, 'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        st.error(f"Gagal memuat resources: {e}")
        return None

# ========== KONFIGURASI STREAMLIT ==========
st.set_page_config(page_title="Analisis Sentimen WhatsApp", page_icon="😊", layout="wide")
st.title("😊 Analisis Sentimen Ulasan WhatsApp")
st.markdown("Menggunakan **LSTM + Rating** untuk menentukan sentimen negatif, netral, atau positif.")

# Load resources
resources = load_resources()
if not resources:
    st.stop()

stemmer, stop_words, model, tokenizer = resources.values()

# Input dari user
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Teks ulasan:", height=150, placeholder="Contoh: Aplikasi ini sangat membantu...")
with col2:
    rating = st.slider("Rating (1-5):", 1, 5, 3)
    analyze = st.button("🔍 Analisis Sentimen", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            # Preprocessing
            cleaned = clean_text(text, stemmer, stop_words)
            # Tokenisasi & padding
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
            # Normalisasi rating
            norm_rating = np.array([[rating / 5.0]], dtype=np.float32)
            # Prediksi
            proba = model.predict({'text_input': padded, 'rating_input': norm_rating}, verbose=0)[0]
            pred = np.argmax(proba)
            sentiment = ["negatif", "netral", "positif"][pred]
            
            # Warna berdasarkan sentimen
            color = {"negatif": "#ef4444", "netral": "#f59e0b", "positif": "#10b981"}
            
            # Tampilkan hasil
            st.markdown(f"""
            <div style="padding:1rem; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
                <h3 style="margin:0">Hasil Sentimen: 
                    <span style="color:{color[sentiment]}">{sentiment.upper()}</span>
                </h3>
                <p><strong>Keyakinan:</strong> {proba[pred]:.2%}</p>
                <p><strong>Rating:</strong> {'⭐'*rating} ({rating}/5)</p>
                <p><strong>Teks bersih:</strong> <i>{cleaned if cleaned else '(kosong)'}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Grafik probabilitas
            st.subheader("📊 Probabilitas per Sentimen")
            fig = px.bar(x=['negatif', 'netral', 'positif'], y=proba,
                         color=['negatif', 'netral', 'positif'],
                         color_discrete_map=color,
                         text=[f"{p:.2%}" for p in proba])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<center><small>Dibangun dengan LSTM - Analisis Sentimen</small></center>", unsafe_allow_html=True)
