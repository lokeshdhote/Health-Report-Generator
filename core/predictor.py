import os
import csv
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from core.recommender import get_recommendations
from core.training import clean_text

MODEL_PATH = 'model/classifier.pkl'

# Load trained model and embedder
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Please train first.")

clf = joblib.load(MODEL_PATH)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def predict_disease(report_text):
    """
    Predicts the disease based on cleaned report text.
    Returns predicted disease and list of recommendations.
    """
    cleaned = clean_text(report_text)
    emb = embedder.encode([cleaned])
    prediction = clf.predict(emb)[0]
    tips = get_recommendations(prediction)
    return prediction, tips

def log_prediction(report_text, disease):
    """
    Logs predictions to CSV with timestamp.
    """
    os.makedirs("data", exist_ok=True)
    with open("data/prediction_log.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_text[:50],
            disease
        ])
