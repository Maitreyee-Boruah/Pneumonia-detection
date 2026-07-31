"""
Chest X-ray Pneumonia Screening — Flask backend.

Loads the trained MobileNetV2 model + metadata.json produced by the
training notebook, exposes a single /predict endpoint that accepts an
uploaded image and returns the predicted class and confidence, and
serves the frontend (templates/index.html).

Research / education tool only — NOT a medical device and NOT a
substitute for professional diagnosis.
"""

import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuration / artifact loading
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Accept either the native Keras format or a converted .h5 file — whichever
# is present in artifacts/ will be used. The native .keras format is what
# the training notebook produces and is the recommended one to deploy with.
MODEL_CANDIDATES = ["best_model.keras", "best_model.h5"]
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _find_model_path():
    for name in MODEL_CANDIDATES:
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No model file found in {ARTIFACTS_DIR}. Expected one of: {MODEL_CANDIDATES}"
    )


with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)

CLASS_NAMES = METADATA["class_names"]              # ["NORMAL", "PNEUMONIA"]
POSITIVE_CLASS = METADATA["positive_class"]         # "PNEUMONIA"
IMAGE_SIZE = tuple(METADATA["image_size"])          # (224, 224)
THRESHOLD = float(METADATA["threshold"])            # e.g. 0.81

MODEL_PATH = _find_model_path()
print(f"Loading model from {MODEL_PATH} ...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Threshold:", THRESHOLD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes):
    """Decode -> RGB -> resize -> float32 in 0-255 (model does its own
    internal preprocess_input scaling, matching the training notebook)."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)  # batch dimension
    return array


def build_result(probability_pneumonia):
    """Turn the raw sigmoid output (probability of PNEUMONIA) into a
    label + a human-facing confidence percentage for whichever class
    was predicted."""
    is_pneumonia = probability_pneumonia >= THRESHOLD
    predicted_label = "PNEUMONIA" if is_pneumonia else "NORMAL"
    confidence = probability_pneumonia if is_pneumonia else (1 - probability_pneumonia)

    return {
        "prediction": predicted_label,
        "confidence": round(float(confidence) * 100, 1),
        "pneumonia_probability": round(float(probability_pneumonia) * 100, 1),
        "threshold_used": round(THRESHOLD * 100, 1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, or BMP image."}), 400

    try:
        file_bytes = file.read()
        image_array = preprocess_image(file_bytes)
    except UnidentifiedImageError:
        return jsonify({"error": "The file could not be read as an image."}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not process the image: {exc}"}), 400

    try:
        raw_prediction = MODEL.predict(image_array, verbose=0)
        probability_pneumonia = float(np.ravel(raw_prediction)[0])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Model inference failed: {exc}"}), 500

    result = build_result(probability_pneumonia)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_path": os.path.basename(MODEL_PATH)})


if __name__ == "__main__":
    # For local development only. Use a production server (e.g. gunicorn)
    # for any real deployment.
    app.run(host="0.0.0.0", port=5000, debug=False)
