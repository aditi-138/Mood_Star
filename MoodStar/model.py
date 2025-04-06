import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

print("📦 Loading dataset...")
df = pd.read_csv("emotions_dataset/training.csv")

# Features and labels
X = df['text']
y = df['label']

print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

print("🧠 Training model...")
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
with open("model/mood_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved in 'model/' folder!")
