import streamlit as st
import pickle
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

model = pickle.load(open("model/mood_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

with open("assets/quotes.json", "r") as f:
    quotes = json.load(f)

GIPHY_API_KEY = "JHSEPOgq7svxVJcJIbT969YfCmkLFtE9"

# Emoji Map
emojis = {
    "0": "😐 Neutral", "1": "😢 Sad", "2": "😠 Angry",
    "3": "😊 Happy", "4": "😴 Tired", "5": "🤩 Excited"
}

def get_gif(emotion_label):
    query = emojis[emotion_label].split()[1]
    url = f"https://api.giphy.com/v1/gifs/search?api_key={GIPHY_API_KEY}&q={query}&limit=1"
    r = requests.get(url).json()
    try:
        return r['data'][0]['images']['original']['url']
    except:
        return None

# 🖼️ App Layout
st.set_page_config(page_title="MoodStar", page_icon="🎭", layout="centered")
st.title("🎭 MoodStar")
st.markdown("**Your mood, your moment – express and explore.**")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Analyze Mood"):
    if user_input.strip():
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        label = str(prediction)
        st.success(f"Detected Emotion: **{emojis[label]}**")

        # Show quote
        st.subheader("💬 Here's something for you:")
        st.write(f"> {random.choice(quotes[label])}")

        # Show GIF
        st.subheader("📸 A GIF to match your mood:")
        gif_url = get_gif(label)
        if gif_url:
            st.image(gif_url)
        else:
            st.write("Sorry, no GIF found right now.")
