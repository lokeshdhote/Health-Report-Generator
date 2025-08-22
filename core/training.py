import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

def train_model(dataset_path='data/sample_dataset.csv', model_path='model/classifier.pkl'):
    """
    Trains a RandomForestClassifier on sentence embeddings from medical reports
    and saves the trained model to disk.
    
    Args:
        dataset_path (str): Path to the CSV dataset containing 'report' and 'disease' columns.
        model_path (str): Path to save the trained model.
    
    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing in the dataset.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Validate dataset columns
    if 'report' not in df.columns or 'disease' not in df.columns:
        raise ValueError("❌ Dataset must contain 'report' and 'disease' columns")

    if df.empty:
        raise ValueError("❌ Dataset is empty. Add data to train the model.")

    # Convert reports into embeddings
    print("🧠 Generating embeddings from reports...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X = embedder.encode(df['report'].tolist(), show_progress_bar=True)
    y = df['disease']

    # Train classifier
    print("🚀 Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save trained model
    joblib.dump(clf, model_path)
    print(f"✅ Model trained and saved to {model_path}")
