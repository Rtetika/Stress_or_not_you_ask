import streamlit as st
import joblib

# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the custom CSS
load_css("styles.css")

# Load the model and CountVectorizer
try:
    with open('stress_model.sav', 'rb') as model_file:
        model = joblib.load(model_file)
    with open('count_vectorizer.sav', 'rb') as cv_file:
        cv = joblib.load(cv_file)
except Exception as e:
    st.error("An error occurred while loading the model and vectorizer files.")
    st.error(str(e))
    st.stop()

# Define the Streamlit app
st.title("Stress Detection App")
st.write("Enter a text to classify it as Stressed or Not Stressed")

# Text input
user_input = st.text_area("Enter text")

# Predict and display result
if st.button("Predict"):
    if user_input:
        try:
            data = cv.transform([user_input]).toarray()
            prediction = model.predict(data)
            st.write(f"The text is classified as: **{prediction[0]}**")
        except Exception as e:
            st.error("An error occurred while making the prediction.")
            st.error(str(e))
    else:
        st.write("Please enter some text to analyze")

if __name__ == '__main__':
    st.write("Streamlit app is running")
