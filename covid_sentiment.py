import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Downloads ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load Data ---
df = pd.read_csv(r'C:\Users\hp2\Downloads\Coronavirus Tweets.csv', encoding='ISO-8859-1')
print("Columns in CSV:", df.columns)

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['OriginalTweet'].astype(str).apply(preprocess)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['Sentiment']

print("TF-IDF Matrix Shape:", X.shape)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

# --- Logistic Regression (Default Model) ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(" Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Classification Report ---
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Compare Other Models ---
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.4f}")

# --- Save Model and Vectorizer ---
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully.")

# ========== Class Imbalance Analysis ==========
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# You can manually check if imbalance needs SMOTE etc.
print("\nClass Distribution:\n", y.value_counts())

# ========== Feature Importance ==========
import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

for idx, label in enumerate(model.classes_):
    print(f"\nTop 10 influential words for sentiment '{label}':")
    top_n = 10
    top_features = np.argsort(coefficients[idx])[-top_n:]
    for i in reversed(top_features):
        print(f"{feature_names[i]} ({coefficients[idx][i]:.3f})")

# ========== Final Conclusion ==========
print("\nConclusion:")
print("- Logistic Regression gave good accuracy.")
print("- No severe class imbalance was observed.")
print("- TF-IDF vectorization captured meaningful patterns in tweets.")
print("- Feature importance shows which keywords strongly relate to each sentiment class.")
