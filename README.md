# PulmoScan

A chest X-ray pneumonia screening UI, paired with a Flask backend that serves
`efficientnetb0-pneumonia-89_71.h5`.

## Setup

1. Put your `efficientnetb0-pneumonia-89_71.h5` file in this same folder (next to `app.py`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the server:
   ```
   python app.py
   ```
4. Open **http://localhost:5000** in your browser.

The frontend automatically detects the backend on load — the pill in the top
right will switch from "demo mode" to "live model connected," and every
"Read film" click will send the image to `/api/predict` and show the model's
real output instead of the simulated one.

## Files

- `index.html` — the lightbox UI (frontend only, no build step)
- `app.py` — Flask server: hosts `index.html` and the `/api/predict` endpoint
- `requirements.txt` — Python dependencies
