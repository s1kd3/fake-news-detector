
import streamlit as st
from src.model import train_model, FAKE, REAL

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article to check whether it is Fake or Real.")

@st.cache_resource
def load_model():
    return train_model()

vectorizer, model = load_model()

user_input = st.text_area("Paste News Article Here")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        X_vec = vectorizer.transform([user_input])
        proba = model.predict_proba(X_vec)[0]

        fake_prob = proba[FAKE]
        real_prob = proba[REAL]

        if real_prob > 0.6:
            st.success(f"✅ REAL news ({real_prob:.2f} confidence)")
        else:
            st.error(f"❌ FAKE news ({fake_prob:.2f} confidence)")

st.markdown("---")
st.caption("Internship Project | Developed by Sayan Kundu")
import streamlit as st
from src.model import train_model, FAKE, REAL

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Check whether a news article is **Fake or Real** using Machine Learning.")

@st.cache_resource
def load_model():
    return train_model()

vectorizer, model = load_model()

user_input = st.text_area("Paste news article here")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("Please enter some news text.")
    else:
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]

        fake_prob = proba[FAKE]
        real_prob = proba[REAL]

        if real_prob >= 0.6:
            st.success(f"✅ REAL News ({real_prob:.2f} confidence)")
        else:
            st.error(f"❌ FAKE News ({fake_prob:.2f} confidence)")

st.markdown("---")
st.caption("Developed by Sayan Kundu")

