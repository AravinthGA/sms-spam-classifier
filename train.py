"""Train and compare SMS spam classifiers, then save the best one.

Usage:
    python train.py                      # uses data/ or downloads the UCI dataset
    python train.py --data path/to/spam.csv
"""
import argparse
import io
import json
import os
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci_dataset():
    """Download the UCI SMS Spam Collection into data/."""
    print("Downloading the UCI SMS Spam Collection ...")
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    archive.extract("SMSSpamCollection", DATA_DIR)
    return os.path.join(DATA_DIR, "SMSSpamCollection")


def find_dataset(explicit_path):
    if explicit_path:
        return explicit_path
    for name in ("SMSSpamCollection", "spam.csv"):
        candidate = os.path.join(DATA_DIR, name)
        if os.path.exists(candidate):
            return candidate
    try:
        return download_uci_dataset()
    except Exception as exc:  # network blocked, URL moved, etc.
        raise SystemExit(
            f"Could not download the dataset ({exc}).\n"
            "Download it manually from https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
            "(or the Kaggle 'SMS Spam Collection Dataset'), put the file in data/, and run again."
        )


def load_data(path):
    """Return a DataFrame with columns: label (0 = ham, 1 = spam) and text."""
    if path.lower().endswith(".csv"):
        # Kaggle layout: v1 = label, v2 = text (plus empty extra columns)
        df = pd.read_csv(path, encoding="latin-1").iloc[:, :2]
    else:
        # UCI layout: tab-separated "label<TAB>text", no header
        df = pd.read_csv(path, sep="\t", header=None, encoding="latin-1", quoting=3)
    df.columns = ["label", "text"]
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    before = len(df)
    df = df.drop_duplicates(subset="text")  # duplicates would leak between train and test
    print(f"Loaded {before} messages, kept {len(df)} after removing duplicates.")
    df["label"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def build_models():
    def tfidf():
        return TfidfVectorizer(lowercase=True, stop_words="english",
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
    }


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }, pred


def save_confusion_matrix(cm, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(np.array(cm), display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train an SMS spam classifier")
    parser.add_argument("--data", help="path to SMSSpamCollection (tab) or spam.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(find_dataset(args.data))
    print(f"Spam share: {df['label'].mean():.1%}")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"])

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics, pred = evaluate(model, x_test, y_test)
        results[name], fitted[name] = metrics, model
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))

    print("\nSummary (spam class, held-out test set)")
    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for name, m in results.items():
        print(f"{name:<22}{m['accuracy']:>10.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}")

    best = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
    print(f"\nBest model by F1: {best}")

    joblib.dump(fitted[best], os.path.join(MODEL_DIR, "spam_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best, "train_size": len(x_train),
                   "test_size": len(x_test), "results": results}, f, indent=2)
    save_confusion_matrix(results[best]["confusion_matrix"], best)
    print(f"Saved model to {os.path.join('model', 'spam_model.joblib')}")


if __name__ == "__main__":
    main()
