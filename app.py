import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #312e81 100%);
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-size: 2.7rem;
    margin: 0 0 .5rem 0;
    font-weight: 800;
}

.hero p {
    color: #d1d5db;
    margin: 0;
    font-size: 1.05rem;
}

.card {
    padding: 1.2rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    background: rgba(128,128,128,.06);
    margin-bottom: 1rem;
}

.result {
    padding: 1.3rem;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.22);
    text-align: center;
}

.small {
    color: #6b7280;
    font-size: .88rem;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 44px;
}
</style>
""", unsafe_allow_html=True)

analyzer = SentimentIntensityAnalyzer()

def analyze_text(text):
    scores = analyzer.polarity_scores(text)
    c = scores["compound"]
    if c >= 0.05:
        label, emoji = "Positive", "😊"
    elif c <= -0.05:
        label, emoji = "Negative", "😞"
    else:
        label, emoji = "Neutral", "😐"
    return label, emoji, scores

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>🧠 Sentiment AI</h1>
    <p>Analyze text sentiment and explore sentiment distributions through an interactive dashboard.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Controls")
    page = st.radio(
        "Navigate",
        ["💬 Text Analyzer", "📊 Dataset Explorer"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Built with Python • VADER • Pandas • Matplotlib • Streamlit")
    st.caption("Portfolio project by Sabeer Mirza")

# ---------- Text Analyzer ----------
if page == "💬 Text Analyzer":
    st.subheader("💬 Text Sentiment Analyzer")
    st.write("Enter a review, comment, or sentence and get an instant sentiment result.")

    examples = {
        "Select an example": "",
        "Positive": "I absolutely loved this product. The quality is amazing!",
        "Neutral": "The product arrived yesterday and works as expected.",
        "Negative": "I am disappointed with the product. The quality is terrible."
    }

    selected = st.selectbox("Try an example", list(examples.keys()))
    default_text = examples[selected]

    text = st.text_area(
        "Your text",
        value=default_text,
        height=150,
        placeholder="Type something like: I really enjoyed this experience!"
    )

    if st.button("✨ Analyze Sentiment", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            label, emoji, scores = analyze_text(text)

            st.markdown(
                f'<div class="result"><div style="font-size:3rem">{emoji}</div>'
                f'<h2 style="margin:.2rem 0">{label}</h2>'
                f'<p class="small">Overall sentiment classification</p></div>',
                unsafe_allow_html=True
            )

            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compound", f"{scores['compound']:.3f}")
            c2.metric("Positive", f"{scores['pos']:.1%}")
            c3.metric("Neutral", f"{scores['neu']:.1%}")
            c4.metric("Negative", f"{scores['neg']:.1%}")

            st.subheader("📈 Score Breakdown")
            score_df = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Score": [scores["pos"], scores["neu"], scores["neg"]]
            }).set_index("Sentiment")
            st.bar_chart(score_df)

            st.caption(
                "VADER compound score ranges from -1 (more negative) to +1 (more positive)."
            )

# ---------- Dataset Explorer ----------
else:
    st.subheader("📊 Dataset Explorer")
    st.write("Upload the CSV used in the original notebook to explore its sentiment distribution.")

    uploaded = st.file_uploader(
        "Upload synthetic_sentiment_analysis_results.csv",
        type=["csv"]
    )

    if uploaded is None:
        st.info("👆 Upload your CSV to start exploring the dataset.")
        st.stop()

    df = pd.read_csv(uploaded)

    before_rows = len(df)
    missing_before = int(df.isnull().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    cleaned = df.dropna().drop_duplicates()

    sentiment_col = next(
        (c for c in cleaned.columns if "sentiment" in c.lower()),
        cleaned.columns[-1]
    )

    counts = cleaned[sentiment_col].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", before_rows)
    c2.metric("Clean Rows", len(cleaned))
    c3.metric("Missing Values", missing_before)
    c4.metric("Duplicates", duplicates_before)

    tab1, tab2, tab3 = st.tabs(["📋 Data", "📊 Charts", "🔎 Summary"])

    with tab1:
        st.write(f"**Sentiment column:** `{sentiment_col}`")
        st.dataframe(cleaned, use_container_width=True, height=400)

    with tab2:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("#### Percentage Distribution")
            fig, ax = plt.subplots(figsize=(6, 5))
            counts.plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.markdown("#### Sentiment Counts")
        summary = counts.rename("Count").to_frame()
        summary["Percentage"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)
        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            cleaned.to_csv(index=False).encode("utf-8"),
            "cleaned_sentiment_data.csv",
            "text/csv",
            use_container_width=True
        )

st.divider()
st.caption("Sentiment AI • Interactive portfolio project")
