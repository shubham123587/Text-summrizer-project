# text_summarizer_app.py
import streamlit as st
from transformers import pipeline

# -------------------------------
# Load summarization model
# -------------------------------
@st.cache_resource
def load_model():
    # Lighter & faster model than facebook/bart-large-cnn
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# -------------------------------
# Function to handle large text
# -------------------------------
def generate_summary(text, summary_length="Medium", max_chunk=1000):
    text = text.replace('\n', ' ')
    if len(text) < 50:
        return "Please enter a longer text."

    # Split text into manageable chunks
    sentences = text.split('. ')
    current_chunk = ''
    chunks = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Summarize each chunk dynamically
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            input_len = len(chunk.split())

            # Length control based on user choice
            if summary_length == "Short":
                max_len = max(40, min(100, input_len // 3))
                min_len = max(20, input_len // 6)
            elif summary_length == "Medium":
                max_len = max(50, min(150, input_len // 2))
                min_len = max(25, input_len // 4)
            else:  # Long
                max_len = max(80, min(200, int(input_len * 0.75)))
                min_len = max(40, int(input_len * 0.3))

            summary = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )[0]['summary_text']

            summaries.append(summary)

        except Exception as e:
            summaries.append(f"[Error summarizing chunk {i+1}: {e}]")

    # Combine all summaries
    final_summary = " ".join(summaries)
    return final_summary


# -------------------------------
# Streamlit GUI
# -------------------------------
st.set_page_config(page_title="AI Text Summarizer", page_icon="🧠", layout="centered")
st.title("🧠 AI-Powered Text Summarizer")
st.write("Paste your long story, essay, or article below and get a meaningful short summary instantly!")

# User input
text = st.text_area("✍️ Enter your text here:", height=300)

# Summary length selection
summary_length = st.radio(
    "📏 Choose summary length:",
    ["Short", "Medium", "Long"],
    horizontal=True
)

# Summarize button
if st.button("Summarize 🪄"):
    with st.spinner("Generating summary... please wait ⏳"):
        result = generate_summary(text, summary_length)
    st.subheader("📄 Summary:")
    st.write(result)

st.markdown("---")
st.markdown("💻 Created by **Shubham** | Powered by 🤗 HuggingFace Transformers")
