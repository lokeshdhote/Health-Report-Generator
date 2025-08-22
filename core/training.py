import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def train_model(dataset_path='data/sample_dataset.csv', model_path='model/classifier.pkl'):
    """
    Trains a RandomForestClassifier on sentence embeddings from medical reports.
    Saves the trained model to disk.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

    # Create model directory if missing
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Validate dataset format
    if 'report' not in df.columns or 'disease' not in df.columns:
        raise ValueError("❌ Dataset must contain 'report' and 'disease' columns")

    # Clean and embed reports
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    X = embedder.encode([clean_text(r) for r in df['report']])
    y = df['disease']

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, model_path)
    print(f"✅ Model trained and saved to {model_path}")
