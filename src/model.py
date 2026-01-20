import pandas as pd
import os
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

nltk.download("stopwords")

FAKE = 0
REAL = 1

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


def train_model():
    df_fake = pd.read_csv(FAKE_PATH)
    df_true = pd.read_csv(TRUE_PATH)

    df_fake["label"] = FAKE
    df_true["label"] = REAL

    df = pd.concat([df_fake, df_true])
    df = df.sample(frac=1, random_state=42)

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return vectorizer, model
