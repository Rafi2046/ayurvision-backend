import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import logging
import uvicorn
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops,
)

# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(title="AyurVision Backend")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# Custom Layer
# ==========================================================

class CBAM(tf.keras.layers.Layer):

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):

        channels = input_shape[-1]

        self.shared_one = tf.keras.layers.Dense(
            channels // self.reduction_ratio,
            activation="relu",
            use_bias=False,
        )

        self.shared_two = tf.keras.layers.Dense(
            channels,
            use_bias=False,
        )

        self.conv_spatial = tf.keras.layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            use_bias=False,
        )

        super().build(input_shape)

    def call(self, inputs):

        avg_pool = tf.reduce_mean(
            inputs,
            axis=[1, 2],
            keepdims=True,
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=[1, 2],
            keepdims=True,
        )

        avg_out = self.shared_two(
            self.shared_one(avg_pool)
        )

        max_out = self.shared_two(
            self.shared_one(max_pool)
        )

        channel = inputs * tf.nn.sigmoid(
            avg_out + max_out
        )

        spatial = tf.concat(
            [
                tf.reduce_mean(
                    channel,
                    axis=-1,
                    keepdims=True,
                ),
                tf.reduce_max(
                    channel,
                    axis=-1,
                    keepdims=True,
                ),
            ],
            axis=-1,
        )

        return channel * self.conv_spatial(spatial)

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "reduction_ratio": self.reduction_ratio
            }
        )

        return config
        
# ==========================================================
# Handcrafted Feature Extractor
# ==========================================================

class HandcraftedExtractor:

    def __init__(self):

        self.lbp_radius = 2
        self.lbp_points = 8 * self.lbp_radius

    # -----------------------------
    # LBP
    # -----------------------------
    def extract_lbp(self, gray):

        lbp = local_binary_pattern(
            gray,
            self.lbp_points,
            self.lbp_radius,
            method="uniform",
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.lbp_points + 3),
            range=(0, self.lbp_points + 2),
        )

        hist = hist.astype("float32")
        hist /= hist.sum() + 1e-7

        return hist

    # -----------------------------
    # GLCM
    # -----------------------------
    def extract_glcm(self, gray):

        glcm = graycomatrix(
            gray,
            distances=[1, 3],
            angles=[
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
            ],
            levels=256,
            symmetric=True,
            normed=True,
        )

        features = []

        for prop in [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        ]:

            features.extend(
                graycoprops(glcm, prop).mean(axis=1)
            )

        return np.array(features)

    # -----------------------------
    # Shape
    # -----------------------------
    def extract_shape(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            return np.zeros(3)

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        circularity = (
            4
            * np.pi
            * area
            / (perimeter * perimeter + 1e-7)
        )

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / (h + 1e-7)

        extent = area / (w * h + 1e-7)

        return np.array(
            [
                circularity,
                aspect_ratio,
                extent,
            ]
        )

    # -----------------------------
    # Color Histogram
    # -----------------------------
    def extract_color(self, hsv):

        hist = []

        for channel in range(3):

            bins = 16

            maximum = 180 if channel == 0 else 256

            h = cv2.calcHist(
                [hsv],
                [channel],
                None,
                [bins],
                [0, maximum],
            )

            hist.extend(h.flatten())

        hist = np.array(hist)

        hist /= hist.sum() + 1e-7

        return hist

    # -----------------------------
    # Final Feature Vector
    # -----------------------------
    def extract_all(self, rgb):

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        )

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        features = np.concatenate(
            [
                self.extract_lbp(gray),
                self.extract_glcm(gray),
                self.extract_shape(mask),
                self.extract_color(hsv),
            ]
        )

        features = np.nan_to_num(features)

        return features.reshape(1, -1)
        # ==========================================================
# Globals
# ==========================================================

MODEL_A_PATH = "mac_best_hybrid_model_new.keras"
MODEL_B_PATH = "mac_best_model_B_new.keras"

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
# Load Models (Only Once)
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

    logger.info("Loading Model A...")

    model_A = tf.keras.models.load_model(
        MODEL_A_PATH,
        custom_objects=custom_objects,
        compile=False,
    )

    logger.info("Model A Loaded")

    logger.info("Loading Model B...")

    model_B = tf.keras.models.load_model(
        MODEL_B_PATH,
        custom_objects=custom_objects,
        compile=False,
    )

    logger.info("Model B Loaded")

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
        "version": "1.0.0",
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

    global model_A, model_B

    logger.info("===================================")
    logger.info("Predict API Called")
    logger.info(f"File: {file.filename}")
    logger.info("===================================")

    try:

        # Load models once
        load_models()

        # Read image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = np.array(image)

        image = cv2.resize(image, (256, 256))

        # CNN input
        img_tensor = tf.cast(
            tf.expand_dims(image, axis=0),
            tf.float32,
        ) / 255.0

        # Handcrafted features
        hc_tensor = hc_extractor.extract_all(image)

        # Predictions
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

        avg = (pA + pB) / 2.0

        top3 = np.argsort(avg)[-3:][::-1]

        results = []

        for i in top3:

            results.append(
                {
                    "class": class_names[i],
                    "confidence": float(avg[i]),
                }
            )

        logger.info(
            f"Prediction: {results[0]['class']}"
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
# Run Server (Local Only)
# ==========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
    )    