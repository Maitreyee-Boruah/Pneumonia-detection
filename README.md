# Chest X-ray Pneumonia Screening — Website

A small Flask website that wraps your trained MobileNetV2 model in a clean,
upload-and-get-a-result interface: upload a chest X-ray, get a NORMAL /
PNEUMONIA call with a confidence percentage.

**Research and education only. This is not a medical device and must not be
used for real diagnosis or treatment decisions.**

## What's in this folder

```
pneumonia_site/
├── app.py                 # Flask backend (loads model, serves site, /predict endpoint)
├── requirements.txt       # Python dependencies
├── convert_to_h5.py       # Optional: convert best_model.keras -> best_model.h5
├── templates/
│   └── index.html         # Frontend (upload UI, result card, all styling/JS inline)
└── artifacts/
    ├── best_model.keras   # Your trained model, copied from your upload
    └── metadata.json      # Class names, image size, decision threshold, test metrics
```

## About the model file format

Your uploaded model is `best_model.keras` — the native Keras 3 format
produced by your training notebook. `app.py` is written to load **either**
`best_model.keras` or `best_model.h5`, whichever it finds in `artifacts/`.

A `.h5` copy could not be generated in this environment because it has no
network access to install TensorFlow. If you specifically need the legacy
`.h5` file (for example, for an older serving stack), run:

```bash
pip install -r requirements.txt
python convert_to_h5.py
```

This reads `artifacts/best_model.keras` and writes `artifacts/best_model.h5`
next to it. Otherwise, just deploy as-is with the `.keras` file — nothing
else needs to change.

## Running it locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the site
python app.py
```

Then open **http://localhost:5000** in your browser.

## How a prediction is made

1. The uploaded image is decoded, converted to RGB, and resized to
   `224x224` (from `metadata.json`).
2. Pixels are kept in the raw `0–255` float range — matching
   `"input_pixels"` in `metadata.json` — because the MobileNetV2
   preprocessing step is already baked into the model itself.
3. The model outputs a single probability that the image shows pneumonia.
4. That probability is compared against the fixed decision threshold from
   training (currently **0.81 / 81%**, chosen to balance sensitivity and
   specificity on the validation set) to produce the final NORMAL /
   PNEUMONIA label, and the confidence shown is how far the probability
   sits from that threshold, on the winning side.

## Deploying it for real use

`app.py` includes a `__main__` block for local development only. For an
actual deployment, run it behind a production WSGI server, for example:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

and put it behind your usual reverse proxy / HTTPS termination (e.g. Nginx,
Render, Railway, Fly.io, etc.).

## Model performance (from your training run)

| Metric | Value |
|---|---|
| Validation balanced accuracy | 93.2% |
| Test accuracy | 79.8% |
| Test sensitivity (catches real pneumonia) | 94.9% |
| Test specificity (correctly clears normal X-rays) | 54.7% |

The model is tuned to lean toward catching pneumonia cases (high
sensitivity) at the cost of more false alarms on normal X-rays (lower
specificity) — worth keeping in mind when interpreting a NORMAL result:
it's the more reliable of the two calls, while PNEUMONIA flags should be
read as "worth a closer look," not a confirmed finding.
