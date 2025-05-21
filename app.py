
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_lemmatize(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Streamlit UI
st.title("📰 Fake News Detector")

st.write("""Enter a news article or headline below, and the model will predict whether it is REAL or FAKE.""")

user_input = st.text_area("📝 Paste news content here:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_lemmatize(user_input)
        vectorized = tfidf.transform([processed])
        prediction = model.predict(vectorized)[0]
        label = "🚫 FAKE News" if prediction == 1 else "✅ REAL News"
        st.subheader(label)

        # Show top words indicating fake or real if user wants
        if st.checkbox("Show top words the model uses"):
            feature_names = tfidf.get_feature_names_out()
            coefs = model.coef_[0]

            top_fake = np.argsort(coefs)[-10:]
            top_real = np.argsort(coefs)[:10]

            st.markdown("### 🔴 Top words indicating FAKE news:")
            for i in reversed(top_fake):
                st.write(f"- {feature_names[i]} ({coefs[i]:.2f})")

            st.markdown("### 🟢 Top words indicating REAL news:")
            for i in top_real:
                st.write(f"- {feature_names[i]} ({coefs[i]:.2f})")
