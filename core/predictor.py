# import joblib
# from sentence_transformers import SentenceTransformer
# from core.recommender import get_recommendations
# import csv
# from datetime import datetime
# import os
# import joblib
# from sentence_transformers import SentenceTransformer
# from core.recommender import get_recommendations
# from core.training import train_model
# import csv
# from datetime import datetime

# MODEL_PATH = 'model/classifier.pkl'


# # Load trained model and embedder
# clf = joblib.load('model/classifier.pkl')
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# def predict_disease(report_text):
#     """
#     Predicts the disease based on the medical report text using the trained model.
#     Returns the predicted disease and a list of recommendations.
#     """
#     emb = embedder.encode([report_text])
#     prediction = clf.predict(emb)[0]
#     suggestions = get_recommendations(prediction)
#     return prediction, suggestions

# def log_prediction(report_text, disease):
#     """
#     Logs the prediction to a CSV file with a timestamp, the first 50 chars of the report, and the predicted disease.
#     """
#     with open("data/prediction_log.csv", "a", newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), report_text[:50], disease])


# # 🔁 Automatically train if model is missing
# if not os.path.exists('model/classifier.pkl'):
#     train_model()

# clf = joblib.load('model/classifier.pkl')
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# def predict_disease(report_text):
#     emb = embedder.encode([report_text])
#     prediction = clf.predict(emb)[0]
#     suggestions = get_recommendations(prediction)
#     return prediction, suggestions

# def log_prediction(report_text, disease):
#     with open("data/prediction_log.csv", "a", newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), report_text[:50], disease])


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
    train_model()

# Load trained model and embedder
clf = joblib.load(MODEL_PATH)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def predict_disease(report_text):
    """
    Predicts the disease based on the medical report text using the trained model.
    Returns the predicted disease and a list of recommendations.
    """
    emb = embedder.encode([report_text])
    prediction = clf.predict(emb)[0]
    suggestions = get_recommendations(prediction)
    return prediction, suggestions

def log_prediction(report_text, disease):
    """
    Logs the prediction to a CSV file with a timestamp, 
    the first 50 characters of the report, and the predicted disease.
    """
    os.makedirs("data", exist_ok=True)  # Ensure 'data' folder exists
    with open("data/prediction_log.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_text[:50],
            disease
        ])
