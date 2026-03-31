# Sentiment Analysis Ulasan WhatsApp (Google Play Store)

## Deskripsi
Project ini bertujuan untuk menganalisis sentimen pengguna terhadap aplikasi WhatsApp berdasarkan ulasan di Google Play Store dengan pendekatan end-to-end mulai dari scraping hingga deployment sederhana (inference).

## Pipeline Project
1. Scraping Data
2. Labeling Sentimen
3. Penyimpanan Dataset
4. Preprocessing Data
5. Training Model
6. Evaluasi Model
7. Inference / Prediksi

---

## 1. Scraping Data
Data diambil langsung dari Google Play Store menggunakan library google-play-scraper.

- Target data: ≥ 10.000 ulasan
- Output:
  - userName
  - score
  - content

---

## 2. Labeling Sentimen
Pelabelan dilakukan secara otomatis dengan kombinasi:
- Keyword-based sentiment
- TextBlob
- Rating (score)

Aturan utama:
- Rating ≥ 4 → cenderung positif
- Rating = 3 → netral
- Rating ≤ 2 → negatif

Output label:
- positif
- netral
- negatif

---

## 3. Penyimpanan Dataset
Dataset hasil scraping dan labeling disimpan dalam format CSV:

WA.csv

Dataset ini akan digunakan untuk tahap training model.

---

## 4. Preprocessing Data
Dilakukan pada notebook modeling.

Langkah:
- Lowercase
- Remove punctuation & angka
- Stopword removal (NLTK)
- Stemming (Sastrawi)

Contoh:
"Sangat Bagus!!!" → "sangat bagus"

---

## 5. Training Model
Dilakukan 3 eksperimen model:

### Skema 1
- Logistic Regression
- TF-IDF
- Split 80:20

### Skema 2
- Random Forest
- Word2Vec
- Split 80:20

### Skema 3
- LSTM (Deep Learning)
- Tokenizer + Padding
- Split 70:30

---

## 6. Evaluasi Model
Metode evaluasi:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Tujuan:
Menentukan model terbaik berdasarkan performa.

---

## 7. Inference (Prediksi)
Model digunakan untuk memprediksi sentimen dari teks baru.

Contoh:
Input:
"aplikasi sangat bagus", rating = 5

Output:
positif

---

## Tools & Library
- Python
- TensorFlow
- Scikit-learn
- Gensim
- NLTK
- Sastrawi
- Matplotlib & Seaborn

---

## Kesimpulan
Pipeline dimulai dari scraping data hingga model mampu melakukan prediksi sentimen secara otomatis. Model LSTM memberikan performa terbaik karena mampu memahami konteks teks.

Project ini dapat digunakan sebagai contoh implementasi NLP di dunia nyata.
