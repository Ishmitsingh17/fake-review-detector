import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Page Configuration
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake & Deceptive Review Detector")
st.markdown("""
This Natural Language Processing (NLP) system evaluates customer feedback and product reviews 
to detect promotional bias, generic bot phrasing, and deceptive reviews.
""")

# 2. Train Model in Memory (Cached)
@st.cache_resource
def train_nlp_detector():
    # Representative corpus of genuine vs bot/promotional deceptive reviews
    training_data = [
        # Deceptive / Bot Reviews (Label = 1)
        ("Best product ever! Must buy, absolutely life changing and perfect in every way!", 1),
        ("Amazing seller, fast shipping! 5 stars! Highly recommend to everyone!", 1),
        ("BUY THIS NOW!! Best purchase I ever made, flawless quality and super cheap!", 1),
        ("Don't think twice, just buy it. Best quality available on the entire internet.", 1),
        ("Click this link for discount: http://bit.ly/deal. Great product 10/10!", 1),
        ("Super fantastic amazing product, I will definitely buy 10 more of these!", 1),
        ("Completely transformed my life overnight. 100% genuine miracle product!", 1),
        ("Best quality on the market. Better than all competitors combined by far!", 1),
        # Genuine Reviews (Label = 0)
        ("The fabric is decent for the price, but the stitching near the zipper feels slightly loose.", 0),
        ("Battery lasts about 6 hours on medium brightness. Good for light office work.", 0),
        ("Sound quality has good bass, though the microphone sounds slightly muffled on calls.", 0),
        ("Arrived two days later than expected. The packaging was a bit dented, but item works.", 0),
        ("Decent laptop for basic browsing. Gets slightly warm during extended video streaming.", 0),
        ("The color is a bit darker than shown in the catalog photos, but it fits true to size.", 0),
        ("Average performance. It does the job, though instructions were poorly translated.", 0),
        ("The camera performs well in daylight, but indoor low-light shots have noticeable noise.", 0)
    ]
    
    texts = [item[0] for item in training_data]
    labels = np.array([item[1] for item in training_data])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(random_state=42)
    model.fit(X, labels)

    return model, vectorizer

with st.spinner("Initializing NLP Classification Engine..."):
    model, vectorizer = train_nlp_detector()

# 3. User Input Interface
st.subheader("Analyze Review")
user_input = st.text_area(
    "Paste customer review or feedback below:",
    placeholder="e.g., The product arrived on time. The build quality is decent, but battery life could be better...",
    height=140
)

# Preset sample test buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Genuine Sample"):
        st.session_state["review_sample"] = "The screen resolution is sharp, but the built-in speaker quality is quite tinny at maximum volume."
with col2:
    if st.button("Load Fake/Bot Sample"):
        st.session_state["review_sample"] = "Best product ever made in history!! 100% buy this immediately, absolutely flawless and incredible!!"

if "review_sample" in st.session_state:
    user_input = st.session_state["review_sample"]

# 4. Prediction Logic
if st.button("Detect Authenticity", type="primary"):
    if not user_input.strip():
        st.warning("Please enter or load a review first.")
    else:
        vec = vectorizer.transform([user_input])
        prob_fake = model.predict_proba(vec)[0][1] * 100
        
        st.subheader("Verdict")
        if prob_fake >= 50.0:
            st.error(f"🚨 **SUSPICIOUS / FAKE REVIEW DETECTED**")
            st.metric(label="Deception Probability", value=f"{prob_fake:.1f}%")
            st.write("Reason: High concentration of superlative marketing adjectives and lack of specific feature critiques.")
        else:
            st.success(f"✅ **GENUINE / AUTHENTIC REVIEW**")
            st.metric(label="Authenticity Confidence", value=f"{100 - prob_fake:.1f}%")
            st.write("Reason: Contains balanced, specific feedback consistent with real user behavior.")
