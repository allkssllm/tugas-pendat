import joblib

# Load model dan vectorizer
model = joblib.load('model_naive_bayes.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Ulasan baru
ulasan_baru = ["aplikasi sering error dan tidak responsif"]

# Preprocessing sederhana
ulasan_baru = [ulasan.lower() for ulasan in ulasan_baru]

# Ubah ke vektor
ulasan_vec = vectorizer.transform(ulasan_baru)

# Prediksi
hasil = model.predict(ulasan_vec)
print("Sentimen:", "Positif" if hasil[0] == 1 else "Negatif")
