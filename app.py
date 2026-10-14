"""Flask web app and JSON API for the SMS spam classifier."""
from flask import Flask, jsonify, render_template, request

from classifier import MAX_LENGTH, classify, load_model

app = Flask(__name__)
model = load_model()

MODEL_MISSING = "The model has not been trained yet. Run `python train.py` and restart the app."


@app.route("/", methods=["GET", "POST"])
def index():
    message, result, error = "", None, None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if model is None:
            error = MODEL_MISSING
        elif not message:
            error = "Type or paste a message to check."
        elif len(message) > MAX_LENGTH:
            error = f"Messages can be up to {MAX_LENGTH} characters."
        else:
            result = classify(model, message)
    return render_template("index.html", message=message, result=result, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify(error=MODEL_MISSING), 503
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify(error="Field 'message' is required."), 400
    if len(message) > MAX_LENGTH:
        return jsonify(error=f"Message must be at most {MAX_LENGTH} characters."), 400
    return jsonify(classify(model, message))


if __name__ == "__main__":
    app.run(debug=False)
