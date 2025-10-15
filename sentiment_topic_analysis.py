"""
Amazon Reviews Sentiment Analysis & Topic Modeling
COMPLETE WORKING CODE FOR GOOGLE COLAB - FIXED ERROR HANDLING
"""

# Step 1: Install required packages
!pip install -q bertopic transformers torch

# Step 2: Import libraries and SET PLOTLY RENDERER FOR COLAB
import pandas as pd
import numpy as np
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import torch
import warnings
warnings.filterwarnings('ignore')

# ⭐ CRITICAL FIX: Set plotly renderer for Google Colab
import plotly.io as pio
pio.renderers.default = 'colab'

print("="*70)
print("AMAZON REVIEWS - SENTIMENT ANALYSIS & TOPIC MODELING")
print("Google Colab Edition - Inline Visualizations")
print("="*70)

# Step 3: Upload your dataset
from google.colab import files
print("\n📂 Upload your amazon_reviews.csv file:")
uploaded = files.upload()

# Step 4: Load and clean data
print("\n📊 Loading and cleaning data...")
df = pd.read_csv("amazon_reviews.csv")
print(f"   ✓ Loaded {len(df)} reviews")

original_count = len(df)
df = df.dropna(subset=["verified_reviews"])
df = df[df["verified_reviews"].str.len() >= 10]
reviews = df["verified_reviews"].astype(str).tolist()

print(f"   ✓ Removed {original_count - len(df)} invalid reviews")
print(f"   ✓ Processing {len(reviews)} valid reviews")
print(f"   ✓ Average review length: {df['verified_reviews'].str.len().mean():.0f} characters\n")

# Step 5: Sentiment Analysis with Batch Processing
print("🔍 Running sentiment analysis with batch processing...")
device = 0 if torch.cuda.is_available() else -1
print(f"   ✓ Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
    max_length=512
)
print("   ✓ Model loaded successfully")

# Batch processing for efficiency
truncated_reviews = [text[:512] for text in df["verified_reviews"].astype(str)]
batch_size = 32 if device == 0 else 16
sentiment_results = []

print(f"   ⏳ Processing {len(truncated_reviews)} reviews in batches of {batch_size}...")

for i in range(0, len(truncated_reviews), batch_size):
    batch = truncated_reviews[i:i+batch_size]
    batch_results = sentiment_pipeline(batch)
    sentiment_results.extend(batch_results)
    
    if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(truncated_reviews):
        print(f"      Progress: {min(i + batch_size, len(truncated_reviews))}/{len(truncated_reviews)} reviews")

df["sentiment"] = [result["label"] for result in sentiment_results]
df["sentiment_score"] = [result["score"] for result in sentiment_results]

print(f"\n   ✅ Sentiment Analysis Complete!")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Sentiment Distribution:")
sentiment_counts = df["sentiment"].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"      • {sentiment}: {count} ({percentage:.1f}%)")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Average Confidence Score: {df['sentiment_score'].mean():.3f}\n")

# Step 6: Topic Modeling with BERTopic
print("🧠 Performing topic modeling with BERTopic...")

try:
    # Configure vectorizer for memory optimization
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.95
    )
    
    # Initialize BERTopic
    print("   ⏳ Initializing BERTopic model...")
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=False,
        low_memory=True,
        vectorizer_model=vectorizer_model,
        min_topic_size=15,
        nr_topics="auto",
        verbose=False
    )
    
    # Fit the model
    print("   ⏳ Fitting model to reviews (this may take 1-3 minutes)...")
    topics, probs = topic_model.fit_transform(reviews)
    
    # ⭐ FIX: Convert topics to numpy array if it's a list
    topics = np.array(topics)
    
    num_topics = len(set(topics)) - 1
    outlier_count = np.sum(topics == -1)  # Now using numpy sum
    
    print(f"\n   ✅ Topic Modeling Complete!")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Discovered Topics: {num_topics}")
    print(f"   Outlier Documents: {outlier_count}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
except Exception as e:
    print(f"   ❌ ERROR in topic modeling: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Display Visualizations Inline in Colab
print("📊 Creating and displaying visualizations...\n")
print("="*70)

# Visualization 1: Bar Chart of Top 10 Topics
print("\n📊 VISUALIZATION 1: Top 10 Topics Bar Chart")
print("="*70)
try:
    fig1 = topic_model.visualize_barchart(top_n_topics=10)
    fig1.show()
    print("✓ Bar chart displayed above\n")
except Exception as e:
    print(f"⚠️ Could not create bar chart: {e}\n")

# Visualization 2: Topic Clusters (2D Map)
print("\n📊 VISUALIZATION 2: Topic Clusters (2D Similarity Map)")
print("="*70)
try:
    fig2 = topic_model.visualize_topics()
    fig2.show()
    print("✓ Topic clusters displayed above")
    print("💡 TIP: Hover over points to see topic keywords!\n")
except Exception as e:
    print(f"⚠️ Could not create cluster map: {e}\n")

# Visualization 3: Topic Hierarchy
print("\n📊 VISUALIZATION 3: Topic Hierarchy")
print("="*70)
try:
    fig3 = topic_model.visualize_hierarchy()
    fig3.show()
    print("✓ Topic hierarchy displayed above\n")
except Exception as e:
    print(f"⚠️ Could not create hierarchy: {e}\n")

# Visualization 4: Topic Heatmap
print("\n📊 VISUALIZATION 4: Topic Similarity Heatmap")
print("="*70)
try:
    fig4 = topic_model.visualize_heatmap()
    fig4.show()
    print("✓ Heatmap displayed above\n")
except Exception as e:
    print(f"⚠️ Could not create heatmap: {e}\n")

# Step 8: Sentiment-Specific Topic Modeling
print("✨ Analyzing topics by sentiment...\n")

positive_reviews = df[df["sentiment"] == "POSITIVE"]["verified_reviews"].tolist()
negative_reviews = df[df["sentiment"] == "NEGATIVE"]["verified_reviews"].tolist()

min_docs = 50

# Positive sentiment topics
if len(positive_reviews) >= min_docs:
    print("="*70)
    print(f"📊 VISUALIZATION 5: Positive Sentiment Topics ({len(positive_reviews)} reviews)")
    print("="*70)
    try:
        print("   ⏳ Analyzing positive reviews...")
        positive_model = BERTopic(
            calculate_probabilities=False,
            low_memory=True,
            min_topic_size=15,
            verbose=False
        )
        positive_model.fit(positive_reviews)
        
        fig_pos = positive_model.visualize_barchart(top_n_topics=5)
        fig_pos.show()
        
        pos_topics = len(set(positive_model.topics_)) - 1
        print(f"   ✓ Positive topics: {pos_topics} discovered")
        print("   ✓ Visualization displayed above\n")
        
    except Exception as e:
        print(f"   ⚠️ Skipped positive topic analysis: {e}\n")
else:
    print(f"⚠️ Only {len(positive_reviews)} positive reviews (need {min_docs}+)\n")

# Negative sentiment topics
if len(negative_reviews) >= min_docs:
    print("="*70)
    print(f"📊 VISUALIZATION 6: Negative Sentiment Topics ({len(negative_reviews)} reviews)")
    print("="*70)
    try:
        print("   ⏳ Analyzing negative reviews...")
        negative_model = BERTopic(
            calculate_probabilities=False,
            low_memory=True,
            min_topic_size=10,
            verbose=False
        )
        negative_model.fit(negative_reviews)
        
        fig_neg = negative_model.visualize_barchart(top_n_topics=5)
        fig_neg.show()
        
        neg_topics = len(set(negative_model.topics_)) - 1
        print(f"   ✓ Negative topics: {neg_topics} discovered")
        print("   ✓ Visualization displayed above\n")
        
    except Exception as e:
        print(f"   ⚠️ Skipped negative topic analysis: {e}\n")
else:
    print(f"⚠️ Only {len(negative_reviews)} negative reviews (need {min_docs}+)\n")

# Step 9: Export Results
print("="*70)
print("💾 Exporting results to CSV files...")
print("="*70)

try:
    # Export 1: Topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("bert_topics.csv", index=False)
    print("   ✓ Saved: bert_topics.csv")
    
    # Export 2: Reviews with analysis
    df["topic"] = topics.tolist()  # Convert numpy array to list for DataFrame
    df.to_csv("reviews_with_analysis.csv", index=False)
    print("   ✓ Saved: reviews_with_analysis.csv")
    
    # Export 3: Detailed topic breakdown
    topic_docs = []
    for topic_id in set(topics):
        if topic_id != -1:
            topic_reviews = df[df["topic"] == topic_id]
            if len(topic_reviews) > 0:  # Safety check
                topic_docs.append({
                    "topic_id": topic_id,
                    "count": len(topic_reviews),
                    "avg_rating": topic_reviews["rating"].mean(),
                    "positive_pct": (topic_reviews["sentiment"] == "POSITIVE").sum() / len(topic_reviews) * 100,
                    "negative_pct": (topic_reviews["sentiment"] == "NEGATIVE").sum() / len(topic_reviews) * 100,
                    "top_keywords": ", ".join([word for word, _ in topic_model.get_topic(topic_id)[:5]])
                })
    
    topic_analysis = pd.DataFrame(topic_docs).sort_values("count", ascending=False)
    topic_analysis.to_csv("topic_analysis_detailed.csv", index=False)
    print("   ✓ Saved: topic_analysis_detailed.csv")
    
    # Export 4: Summary
    summary = {
        "total_reviews": len(df),
        "positive_reviews": len(positive_reviews),
        "negative_reviews": len(negative_reviews),
        "positive_percentage": f"{len(positive_reviews)/len(df)*100:.1f}%",
        "negative_percentage": f"{len(negative_reviews)/len(df)*100:.1f}%",
        "num_topics": num_topics,
        "outlier_documents": int(outlier_count),
        "avg_sentiment_confidence": f"{df['sentiment_score'].mean():.3f}",
        "avg_rating": f"{df['rating'].mean():.2f}"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("analysis_summary.csv", index=False)
    print("   ✓ Saved: analysis_summary.csv")
    
    print("\n✅ All CSV files exported successfully!")
    
except Exception as e:
    print(f"   ⚠️ Export warning: {e}")
    import traceback
    traceback.print_exc()

# Step 10: Final Summary
print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)

print(f"\n📊 RESULTS SUMMARY:")
print(f"   • Total reviews analyzed: {len(df)}")
print(f"   • Positive sentiment: {len(positive_reviews)} ({len(positive_reviews)/len(df)*100:.1f}%)")
print(f"   • Negative sentiment: {len(negative_reviews)} ({len(negative_reviews)/len(df)*100:.1f}%)")
print(f"   • Topics discovered: {num_topics}")
print(f"   • Outlier documents: {int(outlier_count)}")
print(f"   • Average rating: {df['rating'].mean():.2f}/5.0")

print(f"\n📁 CSV FILES CREATED:")
print("   • bert_topics.csv")
print("   • reviews_with_analysis.csv")
print("   • topic_analysis_detailed.csv")
print("   • analysis_summary.csv")

print(f"\n📊 VISUALIZATIONS DISPLAYED ABOVE:")
print("   • Scroll up to see all 4-6 interactive charts!")
print("   • Each chart is interactive - hover, zoom, and click!")

print("\n💡 TIP: To download CSV files to your computer, run:")
print("   from google.colab import files")
print("   files.download('bert_topics.csv')")

print("\n" + "="*70)
