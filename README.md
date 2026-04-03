# 🧠 COVID-19 Tweet Sentiment Classifier

A Machine Learning-powered Streamlit web application that analyzes and predicts the sentiment of COVID-19-related tweets. The app supports both single tweet input and batch prediction using CSV files.

---

## 🚀 Live Demo

👉 https://7isdqztpu6snkhcvevsje8.streamlit.app/

---

## 🌟 Key Highlights

* End-to-end ML project (data → preprocessing → model → deployment)
* Real-world dataset handling and cleaning
* Interactive Streamlit dashboard
* Supports both real-time and batch predictions
* User-friendly interface with filtering and visualization

---

## 📌 Features

* 🔍 Predict sentiment of a single tweet
* 📊 Batch sentiment analysis using CSV upload
* 🎯 Filter predictions by sentiment
* 📈 Interactive sentiment distribution charts
* 💾 Download results as CSV
* ⚡ Robust handling of messy CSV files (encoding, formatting issues)

---

## 🛠️ Technologies Used

* **Machine Learning:** scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest)
* **Data Processing:** Pandas, NumPy, NLTK
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Frontend:** Streamlit
* **Model Saving:** Joblib
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
covid-tweet-sentiment/
│
├── app.py                     # Streamlit application  
├── covid_sentiment.py        # Model training script  
├── sentiment_model.pkl       # Trained ML model  
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer  
├── requirements.txt          # Dependencies  
└── README.md                 # Project documentation  
```

---

## ⚙️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/paridhi-pandey/covid-tweet-sentiment.git
cd covid-tweet-sentiment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. Tweets are preprocessed (cleaning, stopword removal, lemmatization)
2. Text is converted into numerical features using **TF-IDF**
3. Multiple models were experimented with (Logistic Regression, Naive Bayes, SVM, Random Forest), with **Logistic Regression selected as the final model**
4. Predictions are displayed with interactive visualizations and filtering options

---

## 🧪 Supported Input Format

For CSV upload, ensure your file contains one of the following columns:

* `tweet`
* `OriginalTweet`

---

## 📈 Sample Output

* Sentiment Predictions (Positive / Negative / Neutral)
* Filtered Results
* Pie Chart Visualization
* Downloadable CSV

---

## 📌 Future Improvements

* Add deep learning models (LSTM/BERT)
* Real-time Twitter API integration
* Multi-language sentiment analysis
* Enhanced UI/UX improvements

---

## ⭐ Acknowledgements

* Dataset: COVID-19 Tweets Dataset
* Libraries: scikit-learn, Streamlit, NLTK

---

## 👩‍💻 Author

Paridhi Pandey
B.Tech Student | Machine Learning Enthusiast
