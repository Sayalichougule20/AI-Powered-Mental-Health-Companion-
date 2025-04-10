import os
import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import streamlit as st
import random
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import speech_recognition as sr
import io

st.set_page_config(page_title="EmotiCare - Emotion Based AI Chatbot", 
                   page_icon="🎙️", 
                   layout="centered")

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Load BERT model and tokenizer
BERT_MODEL_PATH = "bert_emotion_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    st.success("✅ BERT emotion model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load BERT emotion model: {e}")

# Emotion labels
bert_emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

emotion_responses = {
    "anger": ["I understand. Take a deep breath. Want some help calming down?", "Try some relaxation techniques!"],
    "fear": ["It's okay to feel scared. Want to talk about it?", "Fear is natural. Take a deep breath."],
    "joy": ["That's wonderful! 😊 Let's keep the good vibes going!", "Joy is a beautiful feeling. Celebrate it!"],
    "love": ["That's heartwarming! Spread the love 💖", "Love makes the world brighter. Tell someone you care!"],
    "sadness": ["I'm here for you. Want to talk about it? 💙", "Things will get better. Stay strong!"],
    "surprise": ["Oh wow! That sounds interesting!", "Surprises can be exciting! Want to share more?"]
}

video_links = {
    "anger": "https://youtu.be/66gH1xmXkzI?si=zDv3BIHqQqXYlEqx",
    "fear": "https://youtu.be/AETFvQonfV8?si=h7JWyBwTyYPKwqtc",
    "joy": "https://youtu.be/OcmcptbsvzQ?si=hUQtzH0vRyGV5hmK",
    "love": "https://youtu.be/UAaWoz9wJ_4?si=Qktt7mDUXRmFda5t",
    "sadness": "https://youtu.be/W937gFzsD-c?si=aT3DcRssJRdF0SeH",
    "surprise": "https://youtu.be/PE2GkSgOZMA?si=yZwanX7PC16C73SG"
}

# -------- Functions --------

def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_bytes) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None

def predict_bert_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    top_index = np.argmax(scores)
    return bert_emotion_labels[top_index]

def generate_response(emotion):
    return random.choice(emotion_responses.get(emotion, ["I'm here to chat! 😊"]))

def record_audio(duration=5, samplerate=16000):
    try:
        st.info("🎙️ Recording for {} seconds...".format(duration))
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        wav_file = "temp_audio.wav"
        wav.write(wav_file, samplerate, audio_data)
        return wav_file
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# -------- UI --------

st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right, #FFDEE9, #B5FFFC); font-family: 'Arial', sans-serif; }
    .title { color: #ff4b5c; font-size: 36px; font-weight: bold; text-align: center; }
    .subtitle { color: #333; font-size: 18px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ EmotiCare - Emotion Based AI Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions from your voice or text and receive personalized responses!</p>', unsafe_allow_html=True)

st.sidebar.title("🔍 Choose Input Method")
input_type = st.sidebar.radio("", ["Text", "Voice"])

if input_type == "Text":
    user_text = st.text_input("💬 Type your message:")
    if st.button("Analyze Emotion 🎭"):
        if user_text:
            with st.spinner("🔍 Analyzing..."):
                detected_emotion = predict_bert_emotion(user_text)
                response = generate_response(detected_emotion)
            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
            st.success(f"🤖 Chatbot: {response}")
            if detected_emotion in video_links:
                st.video(video_links[detected_emotion])
        else:
            st.warning("Please type a message.")

elif input_type == "Voice":
    if st.button("Start Recording 🎙️"):
        wav_file = record_audio(duration=5)
        if wav_file:
            with st.spinner("Processing audio..."):
                text = speech_to_text(wav_file)
                if text:
                    detected_emotion = predict_bert_emotion(text)
                    response = generate_response(detected_emotion)
                    st.markdown(f"### 📝 Transcribed Text: _{text}_")
                    st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
                    st.success(f"🤖 Chatbot: {response}")
                    if detected_emotion in video_links:
                        st.video(video_links[detected_emotion])
                else:
                    st.warning("Could not understand the audio. Try again.")
