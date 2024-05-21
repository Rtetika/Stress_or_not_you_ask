import streamlit as st
import joblib

# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the custom CSS
load_css("styles.css")

# Load the model and CountVectorizer
model = joblib.load('stress_model.sav')
cv = joblib.load('count_vectorizer.sav')

# Define the Streamlit app
st.title("Stress Detection")
st.write("Tell us your situation, what you are feeling? Any symptoms you would like to share? We are here for you.")

# Text input
user_input = st.text_area("Enter text")

# Predict and display result
if st.button("Result"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)
        st.write(f"The text is classified as: **{prediction[0]}**")
    else:
        st.write("Please enter some text to analyze")

if __name__ == '__main__':
    st.write("Streamlit app is running")
