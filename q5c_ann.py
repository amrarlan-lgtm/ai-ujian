# Q5c - ANN multi-input (fitur citra + fitur teks + metadata) -> prioritas Rendah/Sedang/Tinggi
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

rng = np.random.default_rng(7)
n = 1500
img_feat  = rng.normal(0, 1, (n, 32))       # ringkasan fitur CNN (hasil PCA dari 1280-d)
text_feat = rng.normal(0, 1, (n, 32))       # ringkasan TF-IDF/embedding (hasil SVD)
meta      = np.c_[rng.integers(0, 4, n), rng.integers(0, 2, n), rng.integers(0, 10, n)]  # jenis, area_publik, #laporan serupa
X = np.hstack([img_feat, text_feat, meta]).astype("float32")
score = 1.2 * img_feat[:, 0] + 1.0 * text_feat[:, 0] + 0.15 * meta[:, 2] + 0.5 * meta[:, 1]
y = np.digitize(score, np.quantile(score, [0.4, 0.75]))        # 0=Rendah 1=Sedang 2=Tinggi

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
sc = StandardScaler().fit(X_tr)
ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-3,
                    early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=7)
ann.fit(sc.transform(X_tr), y_tr)
pred = ann.predict(sc.transform(X_te))
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=["Rendah", "Sedang", "Tinggi"], digits=3))

# ---------------- Versi Keras (multi-input) untuk Google Colab ----------------
# from tensorflow.keras import layers, Model
# in_img  = layers.Input((1280,), name="img_feat")     # keluaran MobileNetV2
# in_txt  = layers.Input((768,),  name="txt_feat")     # embedding IndoBERT / TF-IDF+SVD
# in_meta = layers.Input((6,),    name="meta")         # one-hot jenis, area publik, dsb.
# a = layers.Dense(64, activation="relu")(in_img)
# b = layers.Dense(64, activation="relu")(in_txt)
# x = layers.Concatenate()([a, b, in_meta])
# x = layers.Dense(128, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
# x = layers.Dense(64,  activation="relu")(x); x = layers.Dropout(0.2)(x)
# out = layers.Dense(3, activation="softmax", name="prioritas")(x)
# model = Model([in_img, in_txt, in_meta], out)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit([Xi, Xt, Xm], y, validation_split=0.15, epochs=50, batch_size=32,
#           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
