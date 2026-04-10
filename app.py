import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- High-Tech Clinical Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0b0e14;
        background-image: radial-gradient(#1e293b 0.5px, transparent 0.5px);
        background-size: 24px 24px;
    }

    .block-container {
        padding-top: 5rem;
        max-width: 850px;
    }

    /* Engine Header */
    .engine-header {
        border-bottom: 2px solid #1e293b;
        padding-bottom: 20px;
        margin-bottom: 40px;
    }

    .engine-title {
        font-weight: 800;
        letter-spacing: -2px;
        font-size: 4rem;
        color: #f8fafc;
        line-height: 1;
        margin: 0;
    }

    .engine-subtitle {
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Input Section */
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b !important;
        border-radius: 4px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Precise Button */
    .stButton button {
        background-color: transparent !important;
        color: #3b82f6 !important;
        border: 1px solid #3b82f6 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-color: #60a5fa !important;
        color: #60a5fa !important;
    }

    /* Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 8px;
        background-color: #111827;
        border: 1px solid #1e293b;
        margin-top: 30px;
    }

    .sentiment-positive {
        color: #22c55e;
        font-weight: 800;
        font-size: 2rem;
    }

    .sentiment-negative {
        color: #ef4444;
        font-weight: 800;
        font-size: 2rem;
    }

    .confidence-value {
        font-size: 3rem;
        font-weight: 300;
        color: #f8fafc;
    }

    /* NLTK status */
    .status-tag {
        font-size: 0.75rem;
        color: #475569;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NLTK & Artifacts ---
class LemmatizingTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, text):
        return [self.lemmatizer.lemmatize(t) for t in word_tokenize(text)]

@st.cache_resource
def load_all():
    nltk.download(['stopwords','punkt','wordnet','omw-1.4','punkt_tab'], quiet=True)
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['vectorizer'], data['stop_words']
    except: return None, None, None

model, vectorizer, stop_words = load_all()

# --- Internal Engine Logic ---
def data_cleaning(text):
    text = text.lower()
    # Contract
    text = re.sub(r"n't", " not", text); text = re.sub(r"'s", " is", text)
    text = re.sub(r'http\S+|[^a-z\s]', '', text)
    return ' '.join([w for w in text.split() if w not in stop_words])

def main():
    # Header Section
    st.markdown("""
        <div class="engine-header">
            <h1 class="engine-title">ANALYSIS ENGINE</h1>
            <div class="engine-subtitle">Natural Language Processing Module / IMDB Core</div>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("CORE CRITICAL ERROR: Model artifacts missing from local root.")
        st.stop()

    # Input Section
    review_input = st.text_area("Input Stream", height=200, placeholder="PASTE DATA STREAM HERE...")

    if st.button("Initialize Analysis"):
        if review_input.strip():
            with st.status("Processing input...", expanded=False) as status:
                st.write("Cleaning token streams...")
                cleaned = data_cleaning(review_input)
                st.write("Vectorizing features...")
                vec = vectorizer.transform([cleaned]).toarray()
                st.write("Querying logistic layer...")
                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0]
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Results
            sentiment = 'POSITIVE 😊' if pred == 1 else 'NEGATIVE 😞'
            color_class = 'sentiment-positive' if pred == 1 else 'sentiment-negative'
            val = max(prob)

            st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div>
                            <div style="font-size: 0.75rem; color: #475569; letter-spacing: 2px;">OUTPUT_SENTIMENT</div>
                            <div class="{color_class}">{sentiment}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; color: #475569; letter-spacing: 2px;">PROB_CONFIDENCE</div>
                            <div class="confidence-value">{val*100:.1f}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(val)
        else:
            st.warning("ERROR: Review stream empty.")

    st.markdown("<br><br><span class='status-tag'>[ END OF DATA ]</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()