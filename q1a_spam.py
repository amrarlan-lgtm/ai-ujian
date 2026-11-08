# Q1a - Contoh penerapan AI: Filter Spam Email (Naive Bayes)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Menangkan hadiah uang tunai sekarang klik link ini",
    "Promo gratis pulsa hanya hari ini klik segera",
    "Selamat anda terpilih mendapat hadiah mobil",
    "Jadwal rapat proyek besok jam sembilan pagi",
    "Mohon kirim laporan tugas kuliah sebelum jumat",
    "Undangan seminar kecerdasan buatan di kampus",
]
labels = [1, 1, 1, 0, 0, 0]          # 1 = spam, 0 = bukan spam

vec = CountVectorizer()               # teks -> vektor frekuensi kata
X = vec.fit_transform(emails)
model = MultinomialNB().fit(X, labels)

tests = ["Klik link untuk hadiah gratis sekarang",
         "Tolong kirim jadwal seminar besok"]
for t in tests:
    p = model.predict_proba(vec.transform([t]))[0]
    print(f"{t!r} -> {'SPAM' if p[1] > 0.5 else 'BUKAN SPAM'} (P(spam)={p[1]:.2f})")
