# Q5b - Pipeline NLP: preprocessing + TF-IDF + klasifikasi jenis masalah
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()
SLANG = {"gk": "tidak", "ga": "tidak", "yg": "yang", "udh": "sudah", "rusak2": "rusak", "bgt": "banget"}

def preprocess(text):
    text = text.lower()                                        # case folding
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)              # cleaning
    text = " ".join(SLANG.get(w, w) for w in text.split())     # normalisasi slang
    text = stopword.remove(text)                               # stopword removal
    return stemmer.stem(text)                                  # stemming

data = [
    ("AC di ruang 301 tidak dingin dan bocor air", "AC"),
    ("AC kelas mati total udh seminggu", "AC"),
    ("pendingin ruangan berisik dan menetes", "AC"),
    ("kursi patah kaki nya bahaya diduduki", "Meja/Kursi"),
    ("meja dosen goyang dan retak", "Meja/Kursi"),
    ("bangku mahasiswa rusak banyak", "Meja/Kursi"),
    ("toilet lantai 2 bau dan kran air mati", "Toilet"),
    ("wc mampet dan lantai licin", "Toilet"),
    ("keran wastafel toilet bocor", "Toilet"),
    ("proyektor kelas tidak menyala papan tulis kotor", "Ruang Kelas"),
    ("lampu ruang kelas padam dan kotor", "Ruang Kelas"),
    ("kelas bocor saat hujan lantai basah", "Ruang Kelas"),
]
texts = [preprocess(t) for t, _ in data]; labels = [l for _, l in data]
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000, C=5))
clf.fit(texts, labels)

URGENT = {"bahaya", "korsleting", "api", "bocor", "patah", "mati", "listrik"}
def urgency_score(raw): return sum(w in raw.lower() for w in URGENT)

for r in ["Kursi di kelas 204 patah, bahaya buat mahasiswa", "AC ruang rapat tidak dingin"]:
    p = clf.predict_proba([preprocess(r)])[0]
    print(f"Laporan : {r}\n  Preprocess: {preprocess(r)}\n  Jenis: {clf.classes_[p.argmax()]} ({p.max():.2f}) | skor urgensi kata kunci: {urgency_score(r)}")
