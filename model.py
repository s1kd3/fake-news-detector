import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

FAKE = 0
REAL = 1

# ---------------- PATH SETUP (IMPORTANT) ----------------
# src/model.py → src → project root → data/raw
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
# -------------------------------------------------------

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    return text

def train_model():
    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(f"Fake.csv NOT FOUND at {FAKE_PATH}")

    if not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(f"True.csv NOT FOUND at {TRUE_PATH}")

    df_fake = pd.read_csv(FAKE_PATH)
    df_true = pd.read_csv(TRUE_PATH)

    df_fake["label"] = FAKE
    df_true["label"] = REAL

    df = pd.concat([df_fake, df_true])
    df["text"] = df["text"].apply(preprocess)

    # -------- BALANCE DATASET --------
    df_fake = df[df["label"] == FAKE]
    df_true = df[df["label"] == REAL]

    df_fake_downsampled = resample(
        df_fake,
        replace=False,
        n_samples=len(df_true),
        random_state=42
    )

    df_balanced = pd.concat([df_fake_downsampled, df_true])
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    X = df_balanced["text"]
    y = df_balanced["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return vectorizer, model
