"""
PulmoScan backend
------------------
Serves index.html (the lightbox UI) and a /api/predict endpoint that runs
efficientnetb0-pneumonia-89_71.h5 on an uploaded chest X-ray.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 — the frontend's backend-detection pill
should flip from "demo mode" to "live model connected" automatically.

Model contract (must match how the model was trained — see pnuemonew.ipynb):
    - input: 224x224 RGB, raw 0-255 pixel values. Rescaling/normalization
      layers are baked into the model itself, so do NOT divide by 255 here.
    - output: softmax over 2 classes. class_indices from flow_from_dataframe
      are alphabetical: {"NORMAL": 0, "PNEUMONIA": 1}.
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "efficientnetb0-pneumonia-89_71.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0, index 1 — alphabetical, matches training

app = Flask(__name__, static_folder=None)

print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")  # raw 0-255, model rescales internally
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/predict", methods=["OPTIONS"])
def predict_options():
    # Lets the frontend's lightweight backend-detection ping succeed.
    return ("", 200)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as multipart/form-data under 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        batch = preprocess(file.read())
    except Exception:
        return jsonify({"error": "Could not read that as an image. Try a JPG or PNG."}), 400

    preds = model.predict(batch, verbose=0)[0]  # shape (2,), softmax
    result = {name.lower(): float(prob) for name, prob in zip(CLASS_NAMES, preds)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
