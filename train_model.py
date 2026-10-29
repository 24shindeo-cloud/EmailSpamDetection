
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("emails.csv", encoding="utf-8", encoding_errors="replace")
df.columns = df.columns.astype(str).str.strip()

print("Columns:", df.columns.tolist())

# Identify text and label columns
column_map = {c.lower(): c for c in df.columns}

text_candidates = [
    "message", "text", "email", "email_text",
    "v2", "body", "content"
]
label_candidates = [
    "label", "category", "class", "spam",
    "target", "v1", "type"
]

text_col = next(
    (column_map[c] for c in text_candidates if c in column_map),
    None
)
label_col = next(
    (column_map[c] for c in label_candidates if c in column_map),
    None
)

if text_col is None or label_col is None or text_col == label_col:
    raise ValueError(
        "Could not identify columns. Check the printed column names "
        "and update text_candidates and label_candidates."
    )

# Prepare dataset
df = df[[text_col, label_col]].copy()
df.columns = ["Message", "Category"]
df = df.drop_duplicates()

df["Message"] = df["Message"].fillna("").astype(str).str.strip()
df["Category"] = (
    df["Category"].fillna("").astype(str).str.strip().str.lower()
)

# Standardize labels
label_map = {
    "spam": "spam",
    "1": "spam",
    "junk": "spam",
    "yes": "spam",
    "ham": "ham",
    "0": "ham",
    "not spam": "ham",
    "not_spam": "ham",
    "legitimate": "ham",
    "normal": "ham",
    "no": "ham"
}

df["Category"] = df["Category"].map(
    lambda x: label_map.get(x, x)
)

df = df[
    df["Category"].isin(["spam", "ham"])
    & (df["Message"] != "")
].copy()

if df["Category"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both spam and ham labels."
    )

if df["Category"].value_counts().min() < 2:
    raise ValueError("Each category needs at least two examples.")

X = df["Message"]
y = df["Category"]

print("\nDataset shape:", df.shape)
print("Label counts:\n", y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models to compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

best_model = None
best_name = None
best_f1 = -1

for name, classifier in models.items():
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2)
            )
        ),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    spam_f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Spam F1-score: {spam_f1:.4f}")

    if spam_f1 > best_f1:
        best_f1 = spam_f1
        best_name = name
        best_model = pipeline

# Save trained pipeline
with open("best_spam_email_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model:", best_name)
print("Best spam F1-score:", round(best_f1, 4))
print("Saved: best_spam_email_model.pkl")
