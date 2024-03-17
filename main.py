import threading

from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pyttsx3

text = pyttsx3.init()
ps = PorterStemmer()


def speak(msg):
    try:
        # Use a separate thread for speech synthesis
        def speak_thread():
            text.say(msg)
            text.runAndWait()

        thread = threading.Thread(target=speak_thread)
        thread.start()
    except Exception as e:
        print(f"Error in speak: {str(e)}")


def trasform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    trasformed_sms = trasform_text(input_sms)

    vect_input = tfidf.transform([trasformed_sms])

    result = model.predict(vect_input)[0]

    if result == 1:
        st.header("Spam")
        speak("Spam")
    else:
        st.header("Not spam")
        speak("Not spam")
