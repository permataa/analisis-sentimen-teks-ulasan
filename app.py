import streamlit as st
import pickle
import numpy as np
import re
import time
import plotly.express as px

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
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+|[^\w\s]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Load resources
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
        progress_bar.progress(20)
        
        # Model Logistic Regression
        status_text.text("Memuat model analisis...")
        with open('models/logistic.pkl', 'rb') as f:
            resources['model'] = pickle.load(f)
        progress_bar.progress(40)
        
        # TF-IDF Vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            resources['tfidf'] = pickle.load(f)
        progress_bar.progress(60)

        status_text.text("Sistem siap digunakan!")
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
        "Analisis",
        ["Logistic Regression"],
        index=0
    )
    st.markdown("---")
    show_details = st.checkbox("Tampilkan detail analisis", value=True)
    show_visualization = st.checkbox("Tampilkan visualisasi", value=True)

# Judul
st.title("😊 Analisis Sentimen")
st.markdown("""
<div class="custom-card">
    Analisis sentimen ulasan aplikasi dengan machine learning. Masukkan teks ulasan dan rating,
    sistem akan menganalisis sentimen secara otomatis.
</div>
""", unsafe_allow_html=True)

# Load once
if 'resources' not in st.session_state:
    st.session_state.resources = load_resources()

if st.session_state.resources:
    stemmer = st.session_state.resources['stemmer']
    stop_words = st.session_state.resources['stop_words']
    model = st.session_state.resources['model']
    tfidf = st.session_state.resources['tfidf']

    # Form input
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area("Masukkan teks ulasan:", height=150, placeholder="Contoh: Aplikasi ini sangat membantu...")
        with col2:
            rating = st.slider("Rating (1-5):", 1, 5, 3, 1)
            
            if st.button("🚀 Analisis Sekarang", use_container_width=True):
                if not text.strip():
                    st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
                else:
                    with st.spinner("Menganalisis sentimen..."):
                        loading_placeholder = st.empty()
                        loading_placeholder.image("https://i.gifer.com/ZZ5H.gif", width=100)

                        try:
                            cleaned_text = clean_text(text)
                            text_vector = tfidf.transform([cleaned_text])
                            features = np.hstack([text_vector.toarray(), np.array([[rating/5.0]])])
                            pred = model.predict(features)[0]
                            result = ["negatif", "netral", "positif"][pred]

                            st.session_state.result = {
                                'text': text,
                                'cleaned_text': cleaned_text,
                                'rating': rating,
                                'sentiment': result,
                                'pred_value': pred
                            }

                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                        finally:
                            loading_placeholder.empty()

    # Output
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
                <h2 style="margin: 0;">Hasil Analisis</h2>
                <span style="font-size: 1.5rem; color: {sentiment_color[result['sentiment']]}">
                    {result['sentiment'].upper()}
                </span>
            </div>
            <div style="margin-top: 1rem;">
                <p><strong>Rating:</strong> {rating_stars} ({result['rating']}/5)</p>
                <p><strong>Teks yang dianalisis:</strong> {result['cleaned_text'][:200]}...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_details:
            with st.expander("🔍 Detail Analisis", expanded=True):
                st.write("**Proses Cleaning:**")
                st.code(f"Original: {result['text']}\nCleaned: {result['cleaned_text']}")
                st.write("**Vektor Fitur:**")
                st.write(f"Dimensi TF-IDF: {tfidf.transform([result['cleaned_text']]).shape}")
                st.write(f"Rating ternormalisasi: {result['rating']/5.0:.2f}")

        if show_visualization:
            st.subheader("📊 Visualisasi")
            sentiments = ['negatif', 'netral', 'positif']
            values = [0, 0, 0]
            values[result['pred_value']] = 1

            fig = px.pie(
                names=sentiments,
                values=values,
                color=sentiments,
                color_discrete_map=sentiment_color,
                hole=0.4
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Tingkat Keyakinan Model:**")
            proba = model.predict_proba(features)[0]
            fig2 = px.bar(
                x=sentiments,
                y=proba,
                color=sentiments,
                color_discrete_map=sentiment_color,
                labels={'x': 'Sentimen', 'y': 'Probabilitas'}
            )
            st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<small>
    Sistem analisis sentimen menggunakan model machine learning. 
    Hasil analisis merupakan prediksi otomatis dan dapat mengandung kesalahan.
</small>
""", unsafe_allow_html=True)
