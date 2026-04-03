import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# Ensure required resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load CSV file
try:
    df = pd.read_csv("C:\Users\hp2\Downloads\Coronavirus Tweets.csv", encoding='utf-8', on_bad_lines='skip', quoting=3, sep=',')
except UnicodeDecodeError:
    df = pd.read_csv("C:\Users\hp2\Downloads\Coronavirus Tweets.csv", encoding='ISO-8859-1', on_bad_lines='skip', quoting=3, sep=',')


# Print column names to verify
print("Columns in CSV:", df.columns)

# Preprocessing function
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = tokenizer.tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return " ".join(tokens)

# Apply preprocessing to 'OriginalTweet' column
df['clean_text'] = df['OriginalTweet'].apply(preprocess)

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Define X (input) and y (target)
X = df['clean_text']
y = df['Sentiment']

# 2. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for performance
X_tfidf = vectorizer.fit_transform(X)

# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Show dimensions to confirm
print("TF-IDF Matrix Shape:", X_tfidf.shape)
print("Train set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

# Import required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 2. Predict on test set
y_pred = model.predict(X_test)

# 3. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

