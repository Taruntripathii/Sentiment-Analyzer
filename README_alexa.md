# 🎙️ Amazon Alexa Reviews — Sentiment Analyser

A full-stack ML web app that classifies Amazon Alexa reviews as **Positive** or **Negative**.

Built by **Tarun** · CSE-AI, 2nd Year · GITS Udaipur

---

## 🔍 What it does

- Paste a single review → get an instant sentiment prediction
- Upload a CSV of reviews → get a labelled CSV back + a distribution pie chart

---

## 🛠️ How it works

1. **Text Preprocessing** — removes non-alphabetic characters, lowercases, strips stopwords, applies Porter Stemming
2. **Vectorization** — CountVectorizer (Bag of Words)
3. **Scaling** — MinMaxScaler
4. **Classification** — XGBoost (also trained Decision Tree and Random Forest; XGBoost performed best)

---

## 📦 Tech Stack

- **Backend** — Flask (REST API)
- **Frontend** — Streamlit (`main.py`) + plain HTML (`landing.html`)
- **ML** — scikit-learn, XGBoost, NLTK
- **Dataset** — Amazon Alexa Reviews TSV (~3,000 records)

---

## 🚀 How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Flask API (keep this running)
flask --app api.py run --port=5000

# 3a. Open the HTML frontend
#     Open landing.html in your browser directly

# 3b. Or use the Streamlit frontend
streamlit run main.py
```

---

## 📁 Project Structure

```
├── api.py                              # Flask backend — prediction logic
├── main.py                             # Streamlit frontend
├── landing.html                        # HTML/Tailwind frontend (served by Flask)
├── index.html                          # Standalone HTML version
├── Data_Exploration___Modelling.ipynb  # EDA + model training notebook
├── amazon_alexa.tsv                    # Raw dataset
├── SentimentBulk.csv                   # Sample bulk input file
├── Predictions.csv                     # Sample predictions output
├── Models/
│   ├── model_xgb.pkl                   # XGBoost (primary model)
│   ├── model_rf.pkl                    # Random Forest
│   ├── model_dt.pkl                    # Decision Tree
│   ├── countVectorizer.pkl             # Fitted CountVectorizer
│   └── scaler.pkl                      # Fitted MinMaxScaler
└── requirements.txt
```

---

## 📝 Notes

- Bulk CSV must have a column named **`Sentence`**
- `landing.html` expects Flask at `localhost:5000` — run the API before predicting
- Three models were compared (DT, RF, XGBoost); XGBoost gave best accuracy and is used by default
