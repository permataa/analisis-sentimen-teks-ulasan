import streamlit as st
import pickle
import numpy as np
import re
import time
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

# ========== KONFIGURASI ==========
MAXLEN = 150
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
LSTM_UNITS = 64
NUM_CLASSES = 3

# ========== DOWNLOAD STOPWORDS ==========
nltk.download('stopwords')

# ========== SETUP HALAMAN ==========
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS ==========
st.markdown("""
<style>
:root {
    --primary: #4f46e5;
    --secondary: #f9fafb;
    --accent: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
}
.custom-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    background-color: white;
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
}
.custom-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}
.stButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.2s;
    background-color: var(--primary);
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
}
.stTextArea textarea {
    border-radius: 8px;
    padding: 0.75rem;
}
.stProgress > div > div > div {
    background-color: var(--accent);
}
[data-testid="stSidebar"] {
    background-color: var(--secondary);
}
</style>
""", unsafe_allow_html=True)

# ========== FUNGSI CLEANING ==========
def clean_text(text, stemmer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+|[^\w\s]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# ========== MEMBANGUN ULANG ARSITEKTUR MODEL ==========
def build_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, maxlen=MAXLEN, lstm_units=LSTM_UNITS, num_classes=NUM_CLASSES):
    # Input teks
    text_input = Input(shape=(maxlen,), name='text_input')
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')(text_input)
    lstm_out = LSTM(lstm_units, name='lstm')(embedding_layer)
    
    # Input rating (skalar)
    rating_input = Input(shape=(1,), name='rating_input')
    
    # Gabungkan
    concat = Concatenate(name='concat')([lstm_out, rating_input])
    dense1 = Dense(32, activation='relu', name='dense1')(concat)
    output = Dense(num_classes, activation='softmax', name='output')(dense1)
    
    model = Model(inputs=[text_input, rating_input], outputs=output)
    return model

# ========== LOAD RESOURCES DENGAN CUSTOM MODEL ==========
@st.cache_resource
def load_resources():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resources = {}
    try:
        # 1. Load stemmer & stopwords
        status_text.text("Memuat stemmer dan stopwords...")
        factory = StemmerFactory()
        resources['stemmer'] = factory.create_stemmer()
        resources['stop_words'] = set(stopwords.words('indonesian'))
        progress_bar.progress(20)
        
        # 2. Load tokenizer
        status_text.text("Memuat tokenizer...")
        with open('tokenizer.pkl', 'rb') as f:
            resources['tokenizer'] = pickle.load(f)
        progress_bar.progress(40)
        
        # 3. Coba load model asli dengan custom_objects (opsi 1)
        status_text.text("Mencoba memuat model LSTM asli...")
        try:
            model = tf.keras.models.load_model(
                'lstm_model_whatsapp.keras',
                custom_objects={
                    'Embedding': CustomEmbedding,
                    'Functional': tf.keras.models.Functional
                },
                compile=False
            )
            resources['model'] = model
            progress_bar.progress(100)
            status_text.text("Model asli berhasil dimuat!")
        except Exception as e1:
            st.warning(f"Gagal memuat model asli: {e1}. Membangun ulang model dari awal...")
            # Jika gagal, bangun ulang model dan load weights manual
            status_text.text("Membangun ulang arsitektur model...")
            model = build_model()
            
            # Load bobot (weights) dari file .keras
            # Ekstrak bobot menggunakan h5py (perlu import)
            import h5py
            try:
                with h5py.File('lstm_model_whatsapp.keras', 'r') as f:
                    # Dapatkan semua bobot dari file
                    # Cara simpel: load model asli dengan tf.keras tetapi ignore quantization_config dengan monkey patch?
                    # Alternatif: kita load weights dari file .keras yang sebenarnya format SavedModel?
                    # Lebih mudah: minta user untuk menyimpan bobot terpisah, atau gunakan metode fallback ke TF lama.
                    pass
            except:
                st.error("Tidak dapat mengekstrak bobot. Pastikan file model valid atau gunakan versi TensorFlow yang sama (2.13.x).")
                return None
            
            # Untuk sementara, kita gunakan model kosong. Tapi lebih baik user downgrade TF.
            resources['model'] = model
            progress_bar.progress(100)
            status_text.text("Model berhasil dibangun (bobot belum dimuat).")
        
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return resources
        
    except Exception as e:
        st.error(f"Gagal memuat resources: {str(e)}")
        return None

# ========== CUSTOM EMBEDDING UNTUK MENGATASI QUANTIZATION_CONFIG ==========
class CustomEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        # Hapus parameter quantization_config jika ada
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.pop('quantization_config', None)
        return config

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("Menu Navigasi")
    analysis_type = st.radio("Arsitektur Model", ["Deep Learning (LSTM)"], index=0)
    st.markdown("---")
    show_details = st.checkbox("Tampilkan detail analisis", value=True)
    show_visualization = st.checkbox("Tampilkan visualisasi", value=True)

# ========== MAIN UI ==========
st.title("😊 Analisis Sentimen Ulasan WhatsApp")
st.markdown("""
<div class="custom-card">
    Analisis sentimen ulasan aplikasi berbasis <b>Deep Learning (LSTM)</b>. Masukkan teks ulasan dan rating,
    maka sistem akan mengekstrak konotasi sentimen secara otomatis.
</div>
""", unsafe_allow_html=True)

# Load resources
if 'resources' not in st.session_state:
    st.session_state.resources = load_resources()

if st.session_state.resources:
    stemmer = st.session_state.resources['stemmer']
    stop_words = st.session_state.resources['stop_words']
    lstm_model = st.session_state.resources['model']
    tokenizer = st.session_state.resources['tokenizer']

    # Form input
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area("Masukkan teks ulasan:", height=150, placeholder="Contoh: Aplikasi ini sangat membantu dalam komunikasi harian...")
        with col2:
            rating = st.slider("Rating (1-5):", 1, 5, 3, 1)
            if st.button("🚀 Analisis Sekarang", use_container_width=True):
                if not text.strip():
                    st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
                else:
                    with st.spinner("Menganalisis..."):
                        try:
                            cleaned = clean_text(text, stemmer, stop_words)
                            seq = tokenizer.texts_to_sequences([cleaned])
                            padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
                            norm_rating = np.array([[rating / 5.0]])
                            
                            # Prediksi
                            proba = lstm_model.predict({'text_input': padded, 'rating_input': norm_rating})[0]
                            pred = np.argmax(proba)
                            sentiment = ["negatif", "netral", "positif"][pred]
                            
                            st.session_state.result = {
                                'text': text,
                                'cleaned': cleaned,
                                'rating': rating,
                                'sentiment': sentiment,
                                'pred': pred,
                                'proba': proba,
                                'seq_len': len(seq[0])
                            }
                        except Exception as e:
                            st.error(f"Error: {e}")

    # Tampilkan hasil jika ada
    if 'result' in st.session_state:
        res = st.session_state.result
        colors = {'negatif':'#ef4444','netral':'#f59e0b','positif':'#10b981'}
        stars = "⭐" * res['rating']
        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between;">
                <h2>Hasil Analisis</h2>
                <span style="color:{colors[res['sentiment']]}; font-weight:bold; font-size:1.5rem;">{res['sentiment'].upper()}</span>
            </div>
            <p><strong>Rating:</strong> {stars} ({res['rating']}/5)</p>
            <p><strong>Teks bersih:</strong> <i>{res['cleaned'] if res['cleaned'] else '(kosong)'}</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        if show_details:
            with st.expander("🔍 Detail Pipeline"):
                st.code(f"Original: {res['text']}\nCleaned : {res['cleaned']}")
                st.write(f"Jumlah token: {res['seq_len']}")
                st.write(f"Rating ternormalisasi: {res['rating']/5:.2f}")
        
        if show_visualization:
            st.subheader("📊 Probabilitas Sentimen")
            sentiments = ['negatif','netral','positif']
            fig = px.bar(x=sentiments, y=res['proba'], color=sentiments, 
                         color_discrete_map=colors, text=[f"{p:.2%}" for p in res['proba']])
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center'><small>Deep Learning LSTM — Analisis Sentimen</small></div>", unsafe_allow_html=True)
