# Q5a - Pipeline Image Processing untuk foto fasilitas kampus (OpenCV)
import cv2, numpy as np

def preprocess_image(img_bgr, size=(224, 224)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)        # 1. resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                            # 2. BGR -> RGB
    img = cv2.GaussianBlur(img, (3, 3), 0)                                # 3. reduksi noise
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)                            # 4. CLAHE (kontras)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img.astype("float32") / 255.0                                  # 5. normalisasi [0,1]

def augment(img):                                                         # 6. augmentasi (training)
    if np.random.rand() < 0.5: img = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((112, 112), np.random.uniform(-15, 15), 1.0)
    img = cv2.warpAffine(img, M, (224, 224))
    return np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)

def handcrafted_features(img):                                            # 7. fitur sederhana
    gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten() / gray.size
    return np.concatenate([[edges.mean() / 255, gray.std() / 255], hist])  # 10 dimensi

# --- demo dengan citra sintetis (kursi rusak = banyak garis retak) ---
demo = np.full((480, 640, 3), 180, np.uint8)
cv2.line(demo, (100, 100), (500, 400), (30, 30, 30), 6)
cv2.line(demo, (300, 50), (350, 450), (30, 30, 30), 4)
x = preprocess_image(demo)
print("Shape:", x.shape, "| min/max:", round(float(x.min()), 2), round(float(x.max()), 2))
print("Fitur handcrafted:", np.round(handcrafted_features(x), 3))

# --- ekstraksi fitur CNN (Keras, dijalankan di Colab / lingkungan ber-TensorFlow) ---
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
# vec_image = cnn.predict(preprocess_input(x[None] * 255.0))   # -> (1, 1280)
