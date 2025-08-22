import os
import csv
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer

from core.recommender import get_recommendations
from core.training import train_model

MODEL_PATH = 'model/classifier.pkl'

# 🔁 Automatically train if model is missing
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model not found, training a new model...")
    train_model()

# Load trained model and embedder only once
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}. Retraining...")
    train_model()
    clf = joblib.load(MODEL_PATH)

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
except Exception as e:
    raise RuntimeError(f"❌ Failed to load SentenceTransformer: {e}")

def predict_disease(report_text: str):
    """
    Predicts the disease based on the medical report text.
    Returns:
        - prediction (str): Predicted disease
        - suggestions (list): List of recommendations
    """
    if not report_text.strip():
        raise ValueError("Report text is empty.")
    
    emb = embedder.encode([report_text])
    prediction = clf.predict(emb)[0]
    suggestions = get_recommendations(prediction)
    return prediction, suggestions

def log_prediction(report_text: str, disease: str):
    """
    Logs the prediction to a CSV file with timestamp, first 50 chars of report, and disease.
    """
    os.makedirs("data", exist_ok=True)
    log_file = "data/prediction_log.csv"
    
    try:
        with open(log_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_text[:50],
                disease
            ])
            
    except Exception as e:
        print(f"❌ Failed to log prediction: {e}")
