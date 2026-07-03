import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import uvicorn
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import logging
from fastapi import FastAPI, UploadFile, File
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

app = FastAPI(title="MediLeaf API Local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. CLASSES & EXTRACTORS
# ==========================================
class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        dim = input_shape[-1]
        self.shared_one = tf.keras.layers.Dense(dim // self.reduction_ratio, activation='relu', use_bias=False)
        self.shared_two = tf.keras.layers.Dense(dim, use_bias=False)
        self.conv_spatial = tf.keras.layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', use_bias=False)
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        avg_out = self.shared_two(self.shared_one(tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)))
        max_out = self.shared_two(self.shared_one(tf.reduce_max(inputs, axis=[1, 2], keepdims=True)))
        channel_refined = inputs * tf.nn.sigmoid(avg_out + max_out)
        spatial_concat = tf.concat([tf.reduce_mean(channel_refined, axis=-1, keepdims=True), tf.reduce_max(channel_refined, axis=-1, keepdims=True)], axis=-1)
        return channel_refined * self.conv_spatial(spatial_concat)

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

class HandcraftedExtractor:
    def __init__(self):
        self.lbp_radius = 2
        self.lbp_n_points = 8 * self.lbp_radius

    def extract_lbp(self, gray_img):
        lbp = local_binary_pattern(gray_img, self.lbp_n_points, self.lbp_radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_n_points + 3), range=(0, self.lbp_n_points + 2))
        hist = hist.astype("float")
        return hist / (hist.sum() + 1e-7)

    def extract_haralick_glcm(self, gray_img):
        glcm = graycomatrix(gray_img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        return np.hstack([graycoprops(glcm, prop).mean(axis=1) for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']])

    def extract_shape_features(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return np.zeros(3)
        c = max(contours, key=cv2.contourArea)
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-7))
        _, _, w, h = cv2.boundingRect(c)
        return np.array([circularity, float(w) / (h + 1e-7), float(area) / ((w * h) + 1e-7)])

    def extract_color_histogram(self, hsv_img):
        hist_color = np.concatenate([cv2.calcHist([hsv_img], [i], None, [16], [0, 256 if i>0 else 180]).flatten() for i in range(3)])
        return hist_color / (hist_color.sum() + 1e-7)

    def extract_all(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        feats = np.nan_to_num(np.concatenate([
            self.extract_lbp(gray), 
            self.extract_haralick_glcm(gray), 
            self.extract_shape_features(binary_mask), 
            self.extract_color_histogram(hsv)
        ]))
        return feats.reshape(1, -1)

# ==========================================
# 2. GLOBALS & LAZY LOAD LOGIC
# ==========================================
model_A = None
model_B = None
class_names = ['Akanda', 'Aloe Vera', 'Anshte Lota', 'Aparajita', 'Arjun', 'Ashwagandha', 'Bideshi Lota', 'Devil\'s Backbone', 'Dipto Luchi', 'Joba', 'Kalojira', 'Kori Pata', 'Kumari Lota', 'Lojjaboti', 'Nim', 'Noyontara', 'Pathorkuchi', 'Pudina', 'Roktokorobi', 'Sojne-Moringa', 'Sorpogondha', 'Thankuni', 'Tulsi']
hc_extractor = HandcraftedExtractor()

@app.post("/predict")
async def predict_herb(file: UploadFile = File(...)):
    global model_A, model_B
    try:
        if model_A is None or model_B is None:
            logger.info("Loading Models with compile=False for compatibility...")
            # compile=False দিলে ভার্সন অমিলজনিত এরর আর আসবে না
            model_A = tf.keras.models.load_model('mac_best_hybrid_model_new.keras', custom_objects={'CBAM': CBAM}, compile=False)
            model_B = tf.keras.models.load_model('mac_model_B_saved', custom_objects={'CBAM': CBAM}, compile=False)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_rgb = np.array(image)
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        img_tensor = tf.cast(tf.expand_dims(img_resized, axis=0), tf.float32) / 255.0
        hc_tensor = hc_extractor.extract_all(img_resized)
        
        pA = model_A.predict([img_tensor, hc_tensor], verbose=0)[0]
        pB = model_B.predict([img_tensor, hc_tensor], verbose=0)[0]
        
        avg_probs = (pA + pB) / 2.0
        top_3_indices = np.argsort(avg_probs)[-3:][::-1]
        results = [{"class": class_names[idx], "confidence": float(avg_probs[idx])} for idx in top_3_indices]
            
        return {
            "success": True,
            "primary_prediction": results[0]["class"],
            "primary_confidence": results[0]["confidence"],
            "top_3": results
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
# ==========================================================
# Globals
# ==========================================================

MODEL_A_PATH = "mac_best_hybrid_model_new.keras"
MODEL_B_PATH = "mac_model_B_saved"

model_A = None
model_B = None

custom_objects = {
    "CBAM": CBAM
}

hc_extractor = HandcraftedExtractor()

class_names = [
    "Akanda",
    "Aloe Vera",
    "Anshte Lota",
    "Aparajita",
    "Arjun",
    "Ashwagandha",
    "Bideshi Lota",
    "Devil's Backbone",
    "Dipto Luchi",
    "Joba",
    "Kalojira",
    "Kori Pata",
    "Kumari Lota",
    "Lojjaboti",
    "Nim",
    "Noyontara",
    "Pathorkuchi",
    "Pudina",
    "Roktokorobi",
    "Sojne-Moringa",
    "Sorpogondha",
    "Thankuni",
    "Tulsi",
]


# ==========================================================
# Model Loader
# ==========================================================

def load_models():

    global model_A
    global model_B

    if model_A is not None and model_B is not None:
        return

    logger.info("===================================")
    logger.info("Loading AI Models...")
    logger.info("===================================")

    if not os.path.exists(MODEL_A_PATH):
        raise FileNotFoundError(f"{MODEL_A_PATH} not found")

    if not os.path.exists(MODEL_B_PATH):
        raise FileNotFoundError(f"{MODEL_B_PATH} not found")

    model_A = tf.keras.models.load_model(
        MODEL_A_PATH,
        custom_objects=custom_objects,
        compile=False,
    )

    model_B = tf.keras.models.load_model(
        MODEL_B_PATH,
        custom_objects=custom_objects,
        compile=False,
    )

    logger.info("===================================")
    logger.info("Models Loaded Successfully")
    logger.info("===================================")


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
def home():

    return {
        "status": "ok",
        "message": "AyurVision Backend Running",
        "version": "1.0.0"
    }


@app.get("/test")
def test():

    return {
        "success": True,
        "message": "Backend is working properly."
    }
# ==========================================================
# Prediction API
# ==========================================================

@app.post("/predict")
async def predict_herb(file: UploadFile = File(...)):

    global model_A
    global model_B

    logger.info("===================================")
    logger.info("Predict API Called")
    logger.info(f"Filename : {file.filename}")
    logger.info("===================================")

    try:

        # Load models only once
        load_models()

        # Read uploaded image
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        image = cv2.resize(image, (256, 256))

        # CNN input
        img_tensor = (
            tf.cast(
                tf.expand_dims(image, axis=0),
                tf.float32,
            )
            / 255.0
        )

        # Handcrafted Features
        hc_tensor = hc_extractor.extract_all(image)

        logger.info("Running Model A...")
        pA = model_A.predict(
            [img_tensor, hc_tensor],
            verbose=0,
        )[0]

        logger.info("Running Model B...")
        pB = model_B.predict(
            [img_tensor, hc_tensor],
            verbose=0,
        )[0]

        avg_probs = (pA + pB) / 2

        top3 = np.argsort(avg_probs)[-3:][::-1]

        results = []

        for idx in top3:

            results.append(
                {
                    "class": class_names[idx],
                    "confidence": float(avg_probs[idx]),
                }
            )

        logger.info(
            f"Prediction Finished : {results[0]['class']}"
        )

        return {
            "success": True,
            "primary_prediction": results[0]["class"],
            "primary_confidence": results[0]["confidence"],
            "top_3": results,
        }

    except Exception as e:

        logger.exception("Prediction Failed")

        return {
            "success": False,
            "error": str(e),
        }


# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )