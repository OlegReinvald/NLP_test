from flask import request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from app import app

# Load the pre-trained model and tokenizer
model_name = r"C:\Users\oleja\PycharmProjects\NLP test\models\ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    url = request.form['url']
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        product_names = predict_product_names(text)
        return jsonify(product_names)
    else:
        return "Failed to retrieve the URL"

def predict_product_names(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())
    product_names = []
    for token, prediction in zip(tokens, predictions.squeeze().tolist()):
        if prediction == 1:  # Adjust based on your label mapping
            product_names.append(token)
    return product_names
