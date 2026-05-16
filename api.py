"""
Amazon Alexa Sentiment Analysis - Backend API
Author: Tarun | CSE-AI 2nd Year, GITS Udaipur
"""

from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "API is live. Sentiment analysis service running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

    try:
        if "file" in request.files:
            # Bulk prediction from uploaded CSV
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif "text" in request.json:
            # Single text prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def preprocess_text(text):
    """Clean and stem a single text string."""
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)


def single_prediction(predictor, scaler, cv, text_input):
    cleaned = preprocess_text(text_input)
    X = cv.transform([cleaned]).toarray()
    X_scaled = scaler.transform(X)
    y_pred = predictor.predict_proba(X_scaled).argmax(axis=1)[0]
    return "Positive" if y_pred == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = [preprocess_text(data.iloc[i]["Sentence"]) for i in range(data.shape[0])]

    X = cv.transform(corpus).toarray()
    X_scaled = scaler.transform(X)
    y_preds = predictor.predict_proba(X_scaled).argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if p == 1 else "Negative" for p in y_preds]

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph


if __name__ == "__main__":
    app.run(port=5000, debug=True)
