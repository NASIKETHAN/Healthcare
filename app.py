import streamlit as st
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import tempfile
import os


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') 

# Setting the API key
gemani_api_key = "your key here"

if not gemani_api_key:
    raise ValueError("Please Add API Key")

genai.configure(api_key=gemani_api_key)

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data and model
with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# LLM configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

gmodel = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a chatbot to assist with health-care related queries. The user gives their health related symptoms and the language they understand. Generate the response as a single paragraph in the language provided by the user.",
)

chat_session = gmodel.start_chat(history=[])

# Text preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_intent_response(intent_list, intents):
    if intent_list:
        tag = intent_list[0]["intent"]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "I'm sorry, I don't understand that. Can you rephrase?"

def get_response_from_llm(prompt, language="English"):
    try:
        input_prompt = f"{prompt} language: {language}"
        response = chat_session.send_message(input_prompt)
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print(f"Error with LLM API: {e}")
    return None

# Language to TTS mapping
LANGUAGE_CODE_MAP = {
    "English": "en",
    "Tamil": "ta",
    "Telugu": "te",
    "Hindi": "hi",
    "Malayalam": "ml"
}

# Language to Speech Recognition mapping (with region-specific codes)
LANGUAGE_RECOGNITION_CODE_MAP = {
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Hindi": "hi-IN",
    "Malayalam": "ml-IN"
}

# Text-to-Speech Function
def text_to_speech(text):
    try:
        user_language = st.session_state.get("user_language", "English")
        tts_language = LANGUAGE_CODE_MAP.get(user_language, "en")

        tts = gTTS(text=text, lang=tts_language)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            audio_file = temp_audio.name

        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        os.remove(audio_file)

    except Exception as e:
        st.error("Error generating or playing text-to-speech audio.")
        print(f"TTS Error: {e}")

# Speech-to-Text Function (Updated for multilingual input)
def speech_to_text():
    recognizer = sr.Recognizer()
    user_language = st.session_state.get("user_language", "English")
    recognition_language = LANGUAGE_RECOGNITION_CODE_MAP.get(user_language, "en-IN")

    with sr.Microphone() as source:
        st.info(f"Listening ({user_language})... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio, language=recognition_language)
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your voice.")
        except sr.RequestError as e:
            st.error("Could not request results. Check your internet connection.")
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Health Care Chatbot", page_icon="🧠")

st.title("Health-Care Chatbot")
st.caption("A chatbot to assist you with health related queries. Remember, this is not a substitute for professional advice.")

# Language Selection
language_options = ["English", "Tamil", "Telugu", "Hindi", "Malayalam"]
user_language = st.sidebar.selectbox("Select Language", language_options, index=0)
st.session_state["user_language"] = user_language

# Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Speech Input
if st.button("🎤 Speak"):
    if "voice_input" not in st.session_state or not st.session_state["voice_input"]:
        user_input = speech_to_text()
        st.session_state["voice_input"] = user_input
    else:
        user_input = st.session_state["voice_input"]
else:
    user_input = st.chat_input("How can I assist you today?")
    st.session_state["voice_input"] = None

# Processing input
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    bot_response = get_response_from_llm(user_input, language=user_language)

    if not bot_response:
        predicted_intents = predict_class(user_input)
        bot_response = get_intent_response(predicted_intents, intents)

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    if st.session_state["voice_input"]:
        text_to_speech(bot_response)

    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    st.session_state["voice_input"] = None
