# IMDB Sentiment Analysis Engine

A high-performance sentiment analysis interface built with **Streamlit** and **Scikit-Learn**. This application uses a Logistic Regression model trained on 150,000 IMDB reviews to classify movie critiques as Positive or Negative with high confidence.

## Features
- **Real-time Predictions:** Get instant sentiment analysis for any movie review.
- **Engine Logic:** Custom pipeline including contraction expansion, noise removal, and lemmatization.
- **Clinical UI:** A minimalist, high-tech interface designed for data-centric analysis.
- **Confidence Scoring:** Visual indicators showing the model's certainty.

## Technical Stack
- **Framework:** Streamlit
- **Model:** Logistic Regression (Scikit-Learn)
- **Vectorization:** TF-IDF with custom Lemmatizing Tokenizer
- **NLP Library:** NLTK

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhishek-pandey7/IMDB-Sentiment-Analysis.git
   cd IMDB-Sentiment-Analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: The main Streamlit interface and analysis engine.
- `sentiment_model.pkl`: The serialized model and vectorizer artifacts.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Configured to ignore large datasets and local caches.

---
*Built for clinical-grade precision in natural language processing.*
