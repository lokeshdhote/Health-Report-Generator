# import os
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sentence_transformers import SentenceTransformer

# def train_model(dataset_path='data/sample_dataset.csv', model_path='model/classifier.pkl'):
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"Dataset not found at {dataset_path}")

#     # Load data
#     df = pd.read_csv(dataset_path)
#     if 'report' not in df.columns or 'disease' not in df.columns:
#         raise ValueError("Dataset must contain 'report' and 'disease' columns")

#     # Encode using sentence transformer
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     X = embedder.encode(df['report'].tolist())
#     y = df['disease']

#     # Train classifier
#     clf = RandomForestClassifier()
#     clf.fit(X, y)

#     # Save model
#     joblib.dump(clf, model_path)
#     print(f"✅ Model trained and saved to {model_path}")


import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

def train_model(dataset_path='data/sample_dataset.csv', model_path='model/classifier.pkl'):
    """
    Trains a RandomForestClassifier on sentence embeddings from medical reports
    and saves the trained model to disk.
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

    # Convert reports into embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X = embedder.encode(df['report'].tolist())
    y = df['disease']

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, model_path)
    print(f"✅ Model trained and saved to {model_path}")
