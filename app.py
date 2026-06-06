import streamlit as st
from textblob import TextBlob
from rake_nltk import Rake
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import re
import spacy

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

st.set_page_config(page_title="AeroSafe NLP", page_icon="✈", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/870/870106.png", width=80)
    st.title("AeroSafe NLP")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("✅ Basic Text Stats")
    st.markdown("✅ Sentiment Analysis")
    st.markdown("✅ Keyword Extraction")
    st.markdown("✅ Word Frequency Chart")
    st.markdown("✅ Text Preprocessing")
    st.markdown("✅ File Upload (.txt)")
    st.markdown("✅ Named Entity Recognition")
    st.markdown("---")
    st.markdown("*Built by Gayatri Naidu*")
    st.markdown("*IBS Bachelor Project 2026*")

st.title("✈ AeroSafe NLP Prototype")
st.write("Analyze aviation safety reports using Natural Language Processing.")

st.markdown("### 📂 Input Text")
input_method = st.radio("Choose input method:", ["Type / Paste text", "Upload a .txt file"], horizontal=True)

user_text = ""
if input_method == "Type / Paste text":
    user_text = st.text_area("📝 Enter text to analyze:", height=150,
        placeholder="e.g. Boeing reported a safety incident at London Heathrow on Monday...")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")
        st.success(f"✅ File loaded: **{uploaded_file.name}** ({len(user_text)} characters)")
        st.text_area("📄 File content preview:", value=user_text[:500] + ("..." if len(user_text) > 500 else ""), height=150, disabled=True)

if st.button("🔍 Analyze", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text or upload a file first.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📝 Basic Stats")
            c1, c2, c3 = st.columns(3)
            c1.metric("Characters", len(user_text))
            c2.metric("Words", len(user_text.split()))
            c3.metric("Lines", user_text.count("\n") + 1)

            st.markdown("### 💬 Sentiment")
            polarity = TextBlob(user_text).sentiment.polarity
            if polarity > 0.1:
                st.success(f"😊 Positive (score: {polarity:.2f})")
            elif polarity < -0.1:
                st.error(f"😠 Negative (score: {polarity:.2f})")
            else:
                st.info(f"😐 Neutral (score: {polarity:.2f})")

            st.markdown("### 🔑 Keywords")
            rake = Rake()
            rake.extract_keywords_from_text(user_text)
            keywords = rake.get_ranked_phrases()
            for kw in keywords[:5]:
                st.write(f"🔹 {kw}")

            st.markdown("### 🧠 Named Entities (NER)")
            doc = nlp(user_text)
            entity_colors = {
                "ORG": "🏢", "GPE": "🌍", "PERSON": "👤",
                "DATE": "📅", "TIME": "⏰", "MONEY": "💰",
                "PRODUCT": "📦", "EVENT": "🎯", "LOC": "📍"
            }
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                seen = set()
                for text, label in entities:
                    key = (text.strip().lower(), label)
                    if key not in seen:
                        seen.add(key)
                        icon = entity_colors.get(label, "🔷")
                        st.write(f"{icon} **{text}** — `{label}`")
            else:
                st.info("No named entities found. Try text with organisation names, locations, or dates.")

        with col_right:
            st.markdown("### 📊 Word Frequency")
            stop_words = set(stopwords.words('english'))
            words = re.findall(r'\b[a-zA-Z]{3,}\b', user_text.lower())
            filtered = [w for w in words if w not in stop_words]
            if filtered:
                word_counts = Counter(filtered).most_common(10)
                labels, values = zip(*word_counts)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(labels[::-1], values[::-1], color='steelblue')
                ax.set_xlabel("Frequency")
                ax.set_title("Top Words")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("### 🧹 Preprocessed Text")
            tokens = word_tokenize(user_text.lower())
            clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
            st.code(" | ".join(clean_tokens))