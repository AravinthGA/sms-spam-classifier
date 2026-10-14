"""Shared helper: load the trained model and classify a message."""
import os

import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "spam_model.joblib")
MAX_LENGTH = 1000


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def classify(model, text):
    """Return {'label': 'spam'|'ham', 'spam_probability': float}."""
    spam_probability = float(model.predict_proba([text])[0][1])
    return {
        "label": "spam" if spam_probability >= 0.5 else "ham",
        "spam_probability": round(spam_probability, 4),
    }
