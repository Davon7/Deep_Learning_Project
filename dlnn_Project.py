
import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load the dataset correctly
data = pd.read_csv('IMDB Dataset (1).csv')

# Ensure the dataset has the correct column names
data.columns = ['review', 'sentiment']

# Extract reviews and sentiments
cust = data['review']
sales = data['sentiment']

# Create a new DataFrame
new_data = pd.DataFrame({'Question': cust, 'Answer': sales})

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    if not isinstance(text, str):  # Handle missing values
        return "empty"

    tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text) if word.isalnum()]
    return ' '.join(tokens) if tokens else "empty"

new_data['tokenized Questions'] = new_data['Question'].apply(preprocess_text)

xtrain = new_data['tokenized Questions'].dropna().astype(str).str.strip().tolist()
xtrain = [text for text in xtrain if text.strip()]  # Remove empty strings

# Provide default text if xtrain is empty
if not xtrain:
    xtrain = ["default text"]

# Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
corpus = tfidf_vectorizer.fit_transform(xtrain)

# Define bot responses
bot_greeting = [
    'Hey there! Ready to dive into some music reviews?',
    'Welcome! What song or album are we exploring today?',
    'Hello, music lover! Let us talk tunes. What is on your playlist?',
    'Hi! Looking for a review, recommendation, or just some music talk?',
    'Hey! Let us break down some beats and lyrics. What is on your mind?'
]

bot_farewell = [
    'Catch you later! Keep the music playing.',
    'Thanks for the chat! Come back anytime for more music reviews.',
    'Signing off, but the music never stops! See you soon.',
    'Hope you found your next favorite song! Until next time.',
    'Rock on! Hit me up whenever you need another review.'
]

human_greeting = [
    'Hey bot! What is happening ?',
    'Hello! Got any good music recommendations?',
    'Hi! Ready to talk about some albums?',
    'Hey, I need a review on this song. Can you help?',
    'Hi there! Let us break down some beats.'
]

human_exit = [
    'Alright, I am out! Catch you later.',
    'Thanks! That was fun. See you next time.',
    'Gotta go, but I will be back for more reviews.',
    'Later, bot! Keep those reviews coming.',
    'Bye for now! Time to go listen to some tunes.'
]

# Random greetings & farewells
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell)

# ------------------------------------ STREAMLIT ---------------------------------------------------------

st.markdown("<h1 style='color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #FFC470; text-align: center; font-family: Serif'>Built by DAVID ONYEABO</h4>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.header('Project Background Information', divider=True)
st.write('An organizational chatbot is an AI-powered virtual assistant designed to assist employees, customers, or stakeholders within an organization. These chatbots automate tasks, provide instant responses, and improve efficiency in areas such as customer service, HR, IT support, and internal communications. They use natural language processing (NLP) to understand queries and can be integrated with enterprise systems like CRM, ERP, and HR platforms.')

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
col2.image('pngwing.com (18).png')

# Chatbot interaction
userPrompt = st.chat_input('Ask your Question')
if userPrompt:
    col1.chat_message("ai").write(userPrompt)

    userPrompt = userPrompt.lower()
    if userPrompt in human_greeting:
        col1.chat_message("human").write(random_greeting)
    elif userPrompt in human_exit:
        col1.chat_message("human").write(random_farewell)
    else:
        prodUserInput = preprocess_text(userPrompt)
        vect_user = tfidf_vectorizer.transform([prodUserInput])
        similarity_scores = cosine_similarity(vect_user, corpus)
        most_similar_index = np.argmax(similarity_scores)

        col1.chat_message("human").write(new_data['Answer'].iloc[most_similar_index])





