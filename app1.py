import streamlit as st
import joblib
import pandas as pd

# Load the trained models
tf_idf = joblib.load("Tf_Idf.pkl")
naive_bayes_classifier = joblib.load("nb_classifier.pkl")

# Preprocessing function for user input
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def preprocess(text):
    text = text.strip()
    text = re.sub("<[^>]*>", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

# Streamlit App
st.title("Movie Review Sentiment Analysis")
st.write("This app predicts whether a review is positive or negative.")

# User input
review_input = st.text_area("Enter the movie review:", "")
if st.button("Predict Sentiment"):
    if review_input:
        # Preprocess the input
        processed_input = preprocess(review_input)
        
        # Transform using TF-IDF vectorizer
        transformed_input = tf_idf.transform([processed_input])
        
        # Prediction
        prediction = naive_bayes_classifier.predict(transformed_input)[0]
        
        # Display the result
        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review to analyze.")

# Optionally, display sample data
if st.checkbox("Show sample data"):
    df = pd.read_excel("IMDB_Dataset_sample.xlsx")
    st.dataframe(df.head())
