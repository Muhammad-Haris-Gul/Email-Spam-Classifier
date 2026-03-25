# 🛡️Email Spam Classifier

This project is an end-to-end **Machine Learning** solution designed to identify spam messages with high precision. It transitions from raw data exploration and NLP preprocessing to a live, interactive web application.

---

## 🧠 Core Concepts & Methodology

### 1. Exploratory Data Analysis (EDA)
Before building the model, I analyzed the dataset to understand the underlying patterns:
* **Data Imbalance:** Handled a dataset where "ham" (legit) messages significantly outnumbered "spam."
* **Feature Engineering:** Used **NLTK** to create new features like `num_characters`, `num_words`, and `num_sentences` to analyze if message length correlates with spam.

### 2. Text Preprocessing (NLP Pipeline)
To prepare raw English text for the model, I built a custom `transform_text` function:
* **Tokenization:** Breaking sentences into individual word units.
* **Normalization:** Converting to lowercase and removing special characters/punctuation.
* **Stop Word Removal:** Filtering out "noise" words (e.g., "the", "is") that don't help in classification.
* **Stemming:** Using the **PorterStemmer** to reduce words to their base form (e.g., "loving" -> "love").

### 3. Vectorization (TF-IDF)
I utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors. 
* Unlike simple word counts, TF-IDF rewards unique "signature" words that are characteristic of spam while penalizing common words that appear everywhere.

### 4. Model Selection: Multinomial Naive Bayes
After benchmarking multiple algorithms, **Multinomial Naive Bayes** was selected for deployment.
* **Why?** It is mathematically optimized for discrete features (like word counts) and is famous for its high efficiency and accuracy in text classification tasks.

---

## ⚙️ The Technical Process

1. **Data Cleaning:** Removed duplicates and null values; encoded labels (0 for Ham, 1 for Spam).
2. **Visualization:** Created histograms and heatmaps using **Seaborn** to visualize the differences between Spam and Ham distributions.
3. **Model Pipeline:** Built a consistent pipeline integrating the `TfidfVectorizer` and the `MultinomialNB` classifier.
4. **Deployment:** Developed a frontend using **Streamlit** to allow users to input messages and get real-time predictions.

---

## 📂 Repository Structure

* `Email-Spam-Detection.ipynb` - The full research, EDA, and training workflow.
* `app.py` - The Streamlit Python script for the web interface.
* `model.pkl` - The trained Naive Bayes model.
* `vectorizer.pkl` - The fitted TF-IDF vectorizer.

---

## 🚀 How to Run
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.
