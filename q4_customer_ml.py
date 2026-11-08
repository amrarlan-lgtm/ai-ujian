# Q4 - Customer Behavior Analysis: Clustering (K-Means) + Prediksi Pembelian (Random Forest)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "usia": rng.integers(18, 60, n),
    "recency_hari": rng.integers(1, 365, n),
    "frekuensi_transaksi": rng.poisson(6, n),
    "total_belanja": rng.gamma(2.0, 400_000, n),
    "kategori_favorit": rng.choice(["Fashion", "Elektronik", "Rumah", "Kecantikan"], n),
    "skor_sentimen_ulasan": rng.uniform(-1, 1, n),
})
logit = (-0.012 * df.recency_hari + 0.25 * df.frekuensi_transaksi
         + 0.8 * df.skor_sentimen_ulasan - 0.5)
df["beli_bulan_depan"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

# ---------- A. UNSUPERVISED: segmentasi pelanggan (RFM + K-Means) ----------
rfm = StandardScaler().fit_transform(df[["recency_hari", "frekuensi_transaksi", "total_belanja"]])
for k in range(2, 6):
    print(f"k={k} silhouette={silhouette_score(rfm, KMeans(k, n_init=10, random_state=42).fit_predict(rfm)):.3f}")
df["segmen"] = KMeans(4, n_init=10, random_state=42).fit_predict(rfm)
print(df.groupby("segmen")[["recency_hari", "frekuensi_transaksi", "total_belanja"]].mean().round(0))

# ---------- B. SUPERVISED: prediksi pembelian ----------
X, y = df.drop(columns="beli_bulan_depan"), df["beli_bulan_depan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num = ["usia", "recency_hari", "frekuensi_transaksi", "total_belanja", "skor_sentimen_ulasan"]
prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), ["kategori_favorit"])],
                         remainder="passthrough")
model = Pipeline([("prep", prep),
                  ("clf", RandomForestClassifier(300, class_weight="balanced", random_state=42))])
cv = cross_val_score(model, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1")
print(f"\nCV F1 (5-fold): {cv.mean():.3f} +/- {cv.std():.3f}")
model.fit(X_tr, y_tr)
pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
