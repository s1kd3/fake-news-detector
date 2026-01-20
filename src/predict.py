import pickle
from .preprocessing import preprocess

model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

def predict_news(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return prediction
