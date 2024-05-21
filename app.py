import streamlit as st
import joblib

# Load the model and CountVectorizer
model = joblib.load('stress_model.sav','rb')
cv = joblib.load('count_vectorizer.sav', 'rb')

# Define the Streamlit app
st.title("Stress Detection App")
st.write("Enter a text to classify it as Stressed or Not Stressed")

# Text input
user_input = st.text_area("Enter text")

# Predict and display result
if st.button("Predict"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)
        st.write(f"The text is classified as: **{prediction[0]}**")
    else:
        st.write("Please enter some text to analyze")

