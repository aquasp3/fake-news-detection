from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string
import unicodedata
import os

app = Flask(__name__)

# Load the model from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(parent_dir, "model.pkl")
Model = joblib.load(model_path)

# Preprocessing function
def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def index():
    return render_template("index.html", entered_text='', result='')

@app.route('/', methods=['POST'])
def predict():
    entered_text = request.form.get('txt', '').strip()
    if not entered_text:
        return render_template("index.html", result="❗ Please enter some text.", entered_text='')

    # Normalize and preprocess
    normalized_text = unicodedata.normalize('NFKD', entered_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    cleaned_text = wordpre(normalized_text)
    input_series = pd.Series([cleaned_text])

    try:
        prediction = Model.predict(input_series)[0]
        return render_template("index.html", result=prediction, entered_text=entered_text)
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}", entered_text=entered_text)

if __name__ == "__main__":
    app.run(debug=True)
