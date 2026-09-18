import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")
st.caption("Interactive version of the sentiment-analysis visualization project.")

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text: str):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, scores

# -----------------------------
# Text sentiment analyzer
# -----------------------------
st.header("💬 Analyze Text")

text = st.text_area(
    "Enter a sentence or review:",
    placeholder="Example: I really enjoyed this product!"
)

if st.button("Analyze Sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label, scores = classify_sentiment(text)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", label)
        c2.metric("Compound", f"{scores['compound']:.3f}")
        c3.metric("Positive", f"{scores['pos']:.1%}")
        c4.metric("Negative", f"{scores['neg']:.1%}")

        st.progress(min(max((scores["compound"] + 1) / 2, 0.0), 1.0))
        st.info(
            "VADER compound score ranges from -1 (more negative) to +1 (more positive)."
        )

st.divider()

# -----------------------------
# Dataset visualization
# -----------------------------
st.header("📈 Dataset Visualization")

uploaded_file = st.file_uploader(
    "Upload your synthetic_sentiment_analysis_results.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Upload the CSV above to view the dataset analysis.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Clean data, matching the original notebook's workflow
before_rows = len(df)
missing_before = int(df.isnull().sum().sum())
duplicates_before = int(df.duplicated().sum())

df = df.dropna().drop_duplicates()

# Detect sentiment column, matching the original notebook
sentiment_col = None
for col in df.columns:
    if "sentiment" in col.lower():
        sentiment_col = col
        break

if sentiment_col is None:
    sentiment_col = df.columns[-1]

st.subheader("Dataset Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows after cleaning", len(df))
m2.metric("Columns", len(df.columns))
m3.metric("Missing values removed", missing_before)
m4.metric("Duplicates removed", duplicates_before)

st.write(f"**Sentiment column:** `{sentiment_col}`")

counts = df[sentiment_col].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Sentiment Percentage")
    fig, ax = plt.subplots(figsize=(6, 6))
    counts.plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Sentiment Percentage Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Sentiment Counts")
st.dataframe(
    counts.rename("Count").to_frame(),
    use_container_width=True
)

st.success("🎯 Visualization complete!")
