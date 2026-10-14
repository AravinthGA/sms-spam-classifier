"""Classify a message from the command line.

Usage: python predict.py "Congratulations! You won a free prize, call now"
"""
import sys

from classifier import classify, load_model

if len(sys.argv) < 2:
    raise SystemExit('Usage: python predict.py "your message here"')

model = load_model()
if model is None:
    raise SystemExit("No trained model found. Run: python train.py")

result = classify(model, " ".join(sys.argv[1:]))
print(f"{result['label'].upper()}  (spam probability: {result['spam_probability']:.1%})")
