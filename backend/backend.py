from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from database import get_db_session
from sqlalchemy import text as sql_text
import torch
import joblib
from dotenv import load_dotenv
import os


# Load variable environment
load_dotenv()

# Load Models:
bert_model_path = os.getenv('BERT_MODEL_PATH')
num_labels = 3
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_labels, local_files_only=True)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)
bert_model.eval()

sgd_model_path = os.getenv('SGD_MODEL_PATH')
sgd_model = joblib.load(sgd_model_path)
tfidf_vectorizer_path = os.getenv('TFIDF_PATH')
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
label_encoder_path = os.getenv('LABEL_ENCODER_PATH')
label_encoder = joblib.load(label_encoder_path)

id2label = {0: 'Negative', 1: 'Neutre', 2: 'Positive'}

db = get_db_session()
session = db()
# API:

app = FastAPI()

class TextInput(BaseModel):
    text : str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text

    # Bert Prediction
    inputs = bert_tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()
    bert_pred = id2label[predicted_class_id]

    # --- SGD Prediction ---
    example_tfidf = tfidf_vectorizer.transform([text])
    predicted_encoded = sgd_model.predict(example_tfidf)
    sgd_pred = label_encoder.inverse_transform(predicted_encoded)[0]

    insert_query = sql_text("""
        INSERT INTO predictions (message, bert_prediction, sgd_prediction) VALUES (:message, :bert_pred, :sgd_pred)
    """)

    session.execute(insert_query, {'message': text, 'bert_pred': bert_pred, 'sgd_pred': sgd_pred})
    session.commit()

    return {
        "bert_prediction": bert_pred,
        "sgd_prediction": sgd_pred
    }


# uvicorn backend:app --reload