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

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download stopwords (satu kali saja)
nltk.download('stopwords')

# CSS Custom
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

# Fungsi cleaning text
def clean_text(text, stemmer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+|[^\w\s]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Load resources (Menggunakan @st.cache_resource agar aman & cepat saat refresh)
@st.cache_resource
def load_resources():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resources = {}
    try:
        # Stemmer dan stopwords
        status_text.text("Memuat stemmer dan stopwords...")
        factory = StemmerFactory()
        resources['stemmer'] = factory.create_stemmer()
        resources['stop_words'] = set(stopwords.words('indonesian'))
        progress_bar.progress(30)
        
        # Model Deep Learning LSTM (.h5)
        status_text.text("Memuat model Deep Learning LSTM...")
        resources['model'] = tf.keras.models.load_model('lstm_model_whatsapp.h5')
        progress_bar.progress(70)
        
        # Tokenizer (.pkl) menggantikan TF-IDF
        status_text.text("Memuat objek Tokenizer...")
        with open('tokenizer.pkl', 'rb') as f:
            resources['tokenizer'] = pickle.load(f)
        progress_bar.progress(100)

        status_text.text("Sistem berbasis LSTM siap digunakan!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return resources
    except Exception as e:
        st.error(f"Gagal memuat resources: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("Menu Navigasi")
    analysis_type = st.radio(
        "Arsitektur Model",
        ["Deep Learning (LSTM)"],
        index=0
    )
    st.markdown("---")
    show_details = st.checkbox("Tampilkan detail analisis", value=True)
    show_visualization = st.checkbox("Tampilkan visualisasi", value=True)

# Judul Utama
st.title("😊 Analisis Sentimen Ulasan WhatsApp")
st.markdown("""
<div class="custom-card">
    Analisis sentimen ulasan aplikasi berbasis <b>Deep Learning (LSTM)</b>. Masukkan teks ulasan dan rating,
    maka sistem arsitektur multi-input akan mengekstrak konotasi sentimen secara otomatis.
</div>
""", unsafe_allow_html=True)

# Ambil resources dari session state
if 'resources' not in st.session_state:
    st.session_state.resources = load_resources()

if st.session_state.resources:
    stemmer = st.session_state.resources['stemmer']
    stop_words = st.session_state.resources['stop_words']
    lstm_model = st.session_state.resources['model']
    tokenizer = st.session_state.resources['tokenizer']

    # Form input layout
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
                    with st.spinner("Menganalisis teks dengan jaringan LSTM..."):
                        loading_placeholder = st.empty()
                        
                        try:
                            # 1. Preprocessing teks
                            cleaned_text = clean_text(text, stemmer, stop_words)
                            
                            # 2. Tokenisasi teks & Padding (Sesuai maxlen=150 pada notebook)
                            text_seq = tokenizer.texts_to_sequences([cleaned_text])
                            text_padded = pad_sequences(text_seq, maxlen=150, padding='post', truncating='post')
                            
                            # 3. Normalisasi rating numerik (Sesuai parameter input notebook)
                            normalized_rating = np.array([[rating / 5.0]])
                            
                            # 4. Melakukan Prediksi Multi-input (Menggunakan Dictionary nama layer input)
                            proba = lstm_model.predict({
                                'text_input': text_padded,
                                'rating_input': normalized_rating
                            })[0]
                            
                            # Menentukan kelas indeks [negatif, netral, positif]
                            pred = np.argmax(proba)
                            result = ["negatif", "netral", "positif"][pred]

                            # Simpan ke session state
                            st.session_state.result = {
                                'text': text,
                                'cleaned_text': cleaned_text,
                                'rating': rating,
                                'sentiment': result,
                                'pred_value': pred,
                                'probabilities': proba,
                                'sequence_len': len(text_seq[0])
                            }

                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat pemrosesan: {str(e)}")
                        finally:
                            loading_placeholder.empty()

    # Tampilan Output Komponen Hasil
    if 'result' in st.session_state:
        result = st.session_state.result
        sentiment_color = {
            'negatif': '#ef4444',
            'netral': '#f59e0b',
            'positif': '#10b981'
        }
        rating_stars = "⭐" * result['rating']

        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0;">Hasil Analisis Sentimen</h2>
                <span style="font-size: 1.5rem; font-weight: bold; color: {sentiment_color[result['sentiment']]}">
                    {result['sentiment'].upper()}
                </span>
            </div>
            <div style="margin-top: 1rem;">
                <p><strong>Skor Bintang Pengguna:</strong> {rating_stars} ({result['rating']}/5)</p>
                <p><strong>Kutipan Bersih:</strong> <i>"{result['cleaned_text'] if result['cleaned_text'] else '(Teks kosong setelah cleaning)'}"</i></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tab atau Container Detail Ekspansi
        if show_details:
            with st.expander("🔍 Detail Pipeline & Representasi Fitur", expanded=True):
                st.write("**Transformasi Alur Teks:**")
                st.code(f"Original : {result['text']}\nCleaned  : {result['cleaned_text']}")
                st.write("**Dimensi Data Masukan Keras:**")
                st.write(f"Jumlah Token Kata Terbaca: `{result['sequence_len']}` kata")
                st.write(f"Bentuk Matriks Padding Teks (Shape): `{text_padded.shape}` (Maxlen: 150)")
                st.write(f"Skor Rating Hasil Skala MinMax (Normalisasi): `{result['rating']/5.0:.2f}`")

        # Grafik Distribusi Visualisasi Probabilitas
        if show_visualization:
            st.subheader("📊 Metrik Probabilitas Sentimen")
            sentiments = ['negatif', 'netral', 'positif']
            
            # Membuat distribusi data biner untuk pie chart
            values_pie = [0, 0, 0]
            values_pie[result['pred_value']] = 1

            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.write("**Dominasi Klasifikasi Sentimen:**")
                fig = px.pie(
                    names=sentiments,
                    values=values_pie,
                    color=sentiments,
                    color_discrete_map=sentiment_color,
                    hole=0.4
                )
                fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=250)
                st.plotly_chart(fig, use_container_width=True)
                
            with col_graph2:
                st.write("**Tingkat Keyakinan Jaringan Saraf (Probabilitas):**")
                fig2 = px.bar(
                    x=sentiments,
                    y=result['probabilities'],
                    color=sentiments,
                    color_discrete_map=sentiment_color,
                    labels={'x': 'Kategori Sentimen', 'y': 'Tingkat Keyakinan'},
                    text=[f"{p:.2%}" for p in result['probabilities']]
                )
                fig2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=250)
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)

# Catatan Kaki Aplikasi
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <small style="color: #6b7280;">
        Implementasi Praktis End-to-End Deep Learning LSTM — Laboratorium Informatika.
    </small>
</div>
""", unsafe_allow_html=True)
