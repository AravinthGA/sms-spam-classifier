# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam). It trains two text classifiers on the UCI SMS Spam Collection, compares them, saves the better one, and serves it through a small Flask web app and a JSON API.

## How it works

1. **Load and clean** the dataset, removing duplicate messages so the same text cannot appear in both the training and test sets.
2. **Vectorise** each message with TF-IDF (lowercasing, English stop-word removal, unigrams and bigrams).
3. **Train** two models on an 80/20 stratified split:
   - Multinomial Naive Bayes
   - Logistic Regression (class-weight balanced)
4. **Evaluate** both on the held-out 20%, reporting accuracy, precision, recall and F1 for the spam class.
5. **Save** the model with the best F1 score to `model/spam_model.joblib`.
6. **Serve** it with Flask: a web page and a `POST /api/predict` endpoint.

Precision and recall matter more than accuracy here because spam is the minority class: a model that labels everything "ham" would still look accurate.

## Tech stack

Python, pandas, scikit-learn, joblib, matplotlib, Flask

## Setup

```bash
git clone https://github.com/AravinthGA/sms-spam-classifier.git
cd sms-spam-classifier

python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

`train.py` downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) into `data/` automatically. If the download is blocked, download it manually and either place `SMSSpamCollection` (UCI) or `spam.csv` (Kaggle version) in `data/`, or pass a path:

```bash
python train.py --data path/to/spam.csv
```

It prints a classification report for each model and writes:

- `model/spam_model.joblib` - the trained pipeline
- `model/metrics.json` - metrics for both models
- `model/confusion_matrix.png` - confusion matrix of the chosen model

## Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000, paste a message and click **Check message**.

## Use the API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "WINNER!! Claim your free prize now"}'
```

```json
{"label": "spam", "spam_probability": 0.97}
```

## Command-line prediction

```bash
python predict.py "Hey, are we still meeting for lunch?"
```

## Results

Run `python train.py` and copy your own numbers from the summary table it prints:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Naive Bayes | | | | |
| Logistic Regression | | | | |

## Project structure

```
sms-spam-classifier/
├── train.py          # load data, train and compare models, save the best
├── classifier.py     # shared load/predict helpers
├── app.py            # Flask web app and JSON API
├── predict.py        # command-line predictions
├── templates/        # web page
├── data/             # dataset (downloaded, not committed)
├── model/            # trained model and metrics (generated, not committed)
└── requirements.txt
```

## Limitations and ideas

- The dataset is small (about 5,500 messages) and mostly English from the UK and Singapore, so it can miss spam styles from other regions or newer scams.
- Try adding character n-grams, an SVM, or a transformer-based model, and compare against this baseline.
- Add a pytest suite and a confusion matrix view to the web page.

## Dataset credit

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM Symposium on Document Engineering, 2011.
