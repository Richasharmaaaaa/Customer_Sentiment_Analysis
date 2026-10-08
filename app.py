import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Sentiment Analysis")

st.title("Customer Sentiment Analysis")
st.write("Analyze customer reviews using Machine Learning.")

# Load dataset
df = pd.read_csv("reviews.csv")

st.subheader("Customer Reviews Dataset")
st.dataframe(df)

# Find text and sentiment columns
text_col = None
sentiment_col = None

for col in df.columns:
    if col.lower() in ["review", "reviews", "text", "comment"]:
        text_col = col
    if col.lower() in ["sentiment", "label", "polarity"]:
        sentiment_col = col

if text_col is None:
    text_col = df.columns[0]

if sentiment_col is None:
    sentiment_col = df.columns[1]

# Prepare data
df = df[[text_col, sentiment_col]].dropna()

X = df[text_col].astype(str)
y = df[sentiment_col].astype(str)

# Train ML model
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

st.subheader("Enter Customer Review")

review = st.text_area("Write your review here:")

if st.button("Analyze Sentiment"):

    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    st.success(f"Predicted Sentiment: {prediction}")