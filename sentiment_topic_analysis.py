import pandas as pd
from bertopic import BERTopic
from transformers import pipeline

# Load the dataset
df = pd.read_csv("amazon_review.csv")
reviews = df["reviewText"].dropna().astype(str).tolist()

# Step 1: Sentiment Analysis using HuggingFace Transformers
print("🔍 Running sentiment analysis...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Analyze only the first 512 tokens (safe length for BERT)
df["sentiment"] = df["reviewText"].dropna().astype(str).apply(lambda x: sentiment_pipeline(x[:512])[0]["label"])

# Step 2: Topic Modeling with BERTopic
print("🧠 Fitting BERTopic model...")
topic_model = BERTopic(language="english", verbose=True)
topics, probs = topic_model.fit_transform(reviews)

# Step 3: Visualize Topics
print("📊 Visualizing topics...")
topic_model.visualize_barchart(top_n_topics=10).show()
topic_model.visualize_topics().show()

# Step 4: Topic Modeling by Sentiment
positive_reviews = df[df["sentiment"] == "POSITIVE"]["reviewText"].tolist()
negative_reviews = df[df["sentiment"] == "NEGATIVE"]["reviewText"].tolist()

if positive_reviews:
    print("✨ Positive sentiment topics:")
    positive_model = BERTopic()
    positive_model.fit(positive_reviews)
    positive_model.visualize_barchart(top_n_topics=5).show()

if negative_reviews:
    print("⚠️ Negative sentiment topics:")
    negative_model = BERTopic()
    negative_model.fit(negative_reviews)
    negative_model.visualize_barchart(top_n_topics=5).show()

# Step 5: Export Topics
topic_model.get_topic_info().to_csv("bert_topics.csv", index=False)
print("Sentiment analysis and topic modeling complete. Results saved to bert_topics.csv.")