import streamlit as st
from textblob import TextBlob
from rake_nltk import Rake
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import re
import spacy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd            # for CSV upload [web:311][web:317]
import PyPDF2                  # for PDF text extraction [web:316][web:326]

from sklearn.feature_extraction.text import TfidfVectorizer  # [web:291]
from sklearn.linear_model import LogisticRegression          # [web:292]
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# NLTK downloads (quiet so they don't spam the app)
# --------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# --------------------------------------------------------------------
# 1. CLASSIFIER DATA & MODEL (TF‑IDF + LOGISTIC REGRESSION)
# --------------------------------------------------------------------
@dataclass
class IncidentExample:
    text: str
    category: str


def get_training_data() -> List[IncidentExample]:
    """Small labelled dataset for incident categories."""
    examples = [
        # Weather
        IncidentExample(
            "The aircraft encountered severe turbulence during cruise at FL350. Seat belt signs were activated.",
            "Weather",
        ),
        IncidentExample(
            "Heavy thunderstorms forced a diversion to an alternate airport. Lightning observed near approach path.",
            "Weather",
        ),
        IncidentExample(
            "Severe windshear encountered during final approach. Go-around initiated at decision height.",
            "Weather",
        ),
        IncidentExample(
            "Microburst activity reported by ATIS. Wind changed from strong headwind to strong tailwind within seconds.",
            "Weather",
        ),

        # Bird Strike
        IncidentExample(
            "Bird strike occurred during takeoff roll. Left engine ingested a large bird. Rejected takeoff performed.",
            "Bird Strike",
        ),
        IncidentExample(
            "Flock of birds seen on runway during approach. Crew executed immediate go-around.",
            "Bird Strike",
        ),
        IncidentExample(
            "Multiple gulls struck nose and windshield during rotation. Post-flight inspection revealed minor damage.",
            "Bird Strike",
        ),
        IncidentExample(
            "Crew reported bird impact on right wing during climb through 3000 feet. No abnormal indications.",
            "Bird Strike",
        ),

        # Equipment Failure
        IncidentExample(
            "Hydraulic system failure indicated during climb. Emergency checklist completed. Aircraft landed safely.",
            "Equipment Failure",
        ),
        IncidentExample(
            "Engine oil pressure warning light illuminated shortly after takeoff. Precautionary landing at nearest airport.",
            "Equipment Failure",
        ),
        IncidentExample(
            "Gear retraction failed after takeoff. Crew performed abnormal checklist and extended gear manually.",
            "Equipment Failure",
        ),
        IncidentExample(
            "Autopilot disconnected unexpectedly during approach. Manual flying required in low visibility conditions.",
            "Equipment Failure",
        ),
        IncidentExample(
            "Oxygen system failed during high altitude cruise. Crew initiated emergency descent and diverted.",
            "Equipment Failure",
        ),

        # ATC Issue
        IncidentExample(
            "First officer read back incorrect altitude clearance. Crew did not cross-check. TCAS RA was issued.",
            "ATC Issue",
        ),
        IncidentExample(
            "Controller issued conflicting clearances to two aircraft on parallel approach. TCAS activated.",
            "ATC Issue",
        ),
        IncidentExample(
            "ATC cleared aircraft to wrong runway. Crew queried the clearance and potential conflict was avoided.",
            "ATC Issue",
        ),
        IncidentExample(
            "Tower frequency congestion resulted in missed landing clearance. Readback error not caught.",
            "ATC Issue",
        ),

        # Human Factors
        IncidentExample(
            "Fatigue due to extended duty hours led to missed callout during approach briefing.",
            "Human Factors",
        ),
        IncidentExample(
            "Captain experienced spatial disorientation in IMC during non-precision approach.",
            "Human Factors",
        ),
        IncidentExample(
            "Crew resource management breakdown during high-workload approach in icing conditions.",
            "Human Factors",
        ),
        IncidentExample(
            "Pilot rushed pre-flight checklist after late arrival, resulting in flap configuration error on takeoff.",
            "Human Factors",
        ),

        # Maintenance
        IncidentExample(
            "Maintenance overlooked cracked brake assembly during pre-flight inspection, discovered after landing.",
            "Maintenance",
        ),
        IncidentExample(
            "Hydraulic fitting not torqued to specification during maintenance visit, causing slow leak.",
            "Maintenance",
        ),
        IncidentExample(
            "Incorrect tire pressure recorded by maintenance personnel, requiring replacement before departure.",
            "Maintenance",
        ),
        IncidentExample(
            "Oil cap not secured properly after engine servicing, resulting in oil streaks observed on nacelle.",
            "Maintenance",
        ),
    ]
    return examples


CATEGORY_EMOJI = {
    "Weather": "🌩️",
    "Bird Strike": "🐦",
    "Equipment Failure": "⚙️",
    "ATC Issue": "📡",
    "Human Factors": "🧠",
    "Maintenance": "🔧",
}


@st.cache_resource(show_spinner=True)
def train_classifier() -> Tuple[TfidfVectorizer, LogisticRegression, float]:
    """Train TF‑IDF + Logistic Regression classifier and return model + test accuracy."""
    data = get_training_data()
    texts = [ex.text for ex in data]
    labels = [ex.category for ex in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",          # built-in English stopwords [web:291]
        max_features=500,
        ngram_range=(1, 2),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Simpler config for cloud compatibility: let sklearn choose solver/multi_class [web:292][web:294]
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    clf.fit(X_train_tfidf, y_train)
    accuracy = clf.score(X_test_tfidf, y_test)

    return vectorizer, clf, accuracy


def extract_key_sentence(text: str) -> str:
    """Very simple extractive summary: pick the longest informative sentence."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text.strip()
    scored = [(len(s.split()), s) for s in sentences]
    scored.sort(reverse=True)
    return scored[0][1]


# --------------------------------------------------------------------
# 2. SPACY MODEL (NER)
# --------------------------------------------------------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


nlp = load_spacy_model()


# --------------------------------------------------------------------
# 3. STREAMLIT PAGE CONFIG & SIDEBAR
# --------------------------------------------------------------------
st.set_page_config(page_title="AeroSafe NLP", page_icon="✈", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/870/870106.png", width=80)
    st.title("AeroSafe NLP")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("✅ Incident Category Classifier")
    st.markdown("✅ Basic Text Stats")
    st.markdown("✅ Sentiment Analysis")
    st.markdown("✅ Keyword Extraction")
    st.markdown("✅ Word Frequency Chart")
    st.markdown("✅ Text Preprocessing")
    st.markdown("✅ File Upload (.txt / .csv / .pdf)")
    st.markdown("✅ Named Entity Recognition")
    st.markdown("---")
    st.markdown("*Built by Gayatri Naidu*")
    st.markdown("*IBS Bachelor Project 2026*")

st.title("✈ AeroSafe NLP Prototype")
st.write("Analyze aviation safety reports using Natural Language Processing.")

# --------------------------------------------------------------------
# 4. INPUT SECTION (TEXT / TXT / CSV / PDF)
# --------------------------------------------------------------------
st.markdown("### 📂 Input Text")
input_method = st.radio(
    "Choose input method:",
    ["Type / Paste text", "Upload a .txt file", "Upload a .csv file", "Upload a .pdf file"],
    horizontal=True,
)

user_text = ""

if input_method == "Type / Paste text":
    user_text = st.text_area(
        "📝 Enter text to analyze:",
        height=150,
        placeholder="e.g. Boeing reported a safety incident at London Heathrow on Monday...",
    )

elif input_method == "Upload a .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")
        st.success(f"✅ File loaded: **{uploaded_file.name}** ({len(user_text)} characters)")
        st.text_area(
            "📄 File content preview:",
            value=user_text[:500] + ("..." if len(user_text) > 500 else ""),
            height=150,
            disabled=True,
        )

elif input_method == "Upload a .csv file":
    uploaded_file = st.file_uploader("Upload a .csv file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # pandas reads directly from UploadedFile [web:311][web:317]
        st.success(f"✅ CSV loaded: **{uploaded_file.name}** ({len(df)} rows)")
        text_column = st.selectbox("Select the column with report text:", df.columns)
        # Combine selected column into one text block
        user_text = "\n".join(df[text_column].astype(str).tolist())
        st.text_area(
            "📄 Sample of combined text:",
            value=user_text[:500] + ("..." if len(user_text) > 500 else ""),
            height=150,
            disabled=True,
        )

elif input_method == "Upload a .pdf file":
    uploaded_file = st.file_uploader("Upload a .pdf file", type=["pdf"])
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)  # [web:316][web:326]
        text_chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
        user_text = "\n".join(text_chunks)
        st.success(f"✅ PDF loaded: **{uploaded_file.name}**")
        st.text_area(
            "📄 Extracted text preview:",
            value=user_text[:500] + ("..." if len(user_text) > 500 else ""),
            height=150,
            disabled=True,
        )

# Load classifier (cached, so trained only once)
with st.spinner("Loading classifier (first run only)…"):
    vectorizer, clf, test_accuracy = train_classifier()

# --------------------------------------------------------------------
# 5. ANALYZE BUTTON
# --------------------------------------------------------------------
if st.button("🔍 Analyze", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text or upload a file first.")
    else:
        text_clean = user_text.strip()

        # ==========================
        # 5.1 CLASSIFICATION SECTION
        # ==========================
        st.markdown("## 1️⃣ Incident Category Classifier")

        X_input = vectorizer.transform([text_clean])
        proba = clf.predict_proba(X_input)[0]
        classes = clf.classes_
        predicted_index = int(np.argmax(proba))
        predicted_category = classes[predicted_index]
        confidence = proba[predicted_index] * 100

        clf_col_left, clf_col_right = st.columns([1, 2])

        with clf_col_left:
            emoji = CATEGORY_EMOJI.get(predicted_category, "✈️")
            st.success(f"**Predicted Category: {emoji} {predicted_category}**")
            st.write(f"**Confidence:** {confidence:.1f}%")

        with clf_col_right:
            st.markdown("**Category probabilities:**")
            sorted_items = sorted(zip(classes, proba), key=lambda x: -x[1])
            for cls, p in sorted_items:
                em = CATEGORY_EMOJI.get(cls, "•")
                st.markdown(f"{em} **{cls}** — {p*100:.1f}%")
                st.progress(float(p))

        st.markdown("### 📝 Key Insight from Report")
        key_sentence = extract_key_sentence(text_clean)
        st.info(f"“{key_sentence}”")

        st.markdown("---")
        st.markdown("## 2️⃣ Advanced NLP Analysis")

        # ==========================
        # 5.2 EXISTING ANALYSIS (LEFT)
        # ==========================
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📝 Basic Stats")
            c1, c2, c3 = st.columns(3)
            c1.metric("Characters", len(text_clean))
            c2.metric("Words", len(text_clean.split()))
            c3.metric("Lines", text_clean.count("\n") + 1)

            st.markdown("### 💬 Sentiment")
            polarity = TextBlob(text_clean).sentiment.polarity
            if polarity > 0.1:
                st.success(f"😊 Positive (score: {polarity:.2f})")
            elif polarity < -0.1:
                st.error(f"😠 Negative (score: {polarity:.2f})")
            else:
                st.info(f"😐 Neutral (score: {polarity:.2f})")

            st.markdown("### 🔑 Keywords")
            rake = Rake()
            rake.extract_keywords_from_text(text_clean)
            keywords = rake.get_ranked_phrases()
            for kw in keywords[:5]:
                st.write(f"🔹 {kw}")

            st.markdown("### 🧠 Named Entities (NER)")
            doc = nlp(text_clean)
            entity_colors = {
                "ORG": "🏢",
                "GPE": "🌍",
                "PERSON": "👤",
                "DATE": "📅",
                "TIME": "⏰",
                "MONEY": "💰",
                "PRODUCT": "📦",
                "EVENT": "🎯",
                "LOC": "📍",
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
                st.info(
                    "No named entities found. Try text with organisation names, locations, or dates."
                )

        # ==========================
        # 5.3 EXISTING ANALYSIS (RIGHT)
        # ==========================
        with col_right:
            st.markdown("### 📊 Word Frequency")
            stop_words = set(stopwords.words("english"))
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text_clean.lower())
            filtered = [w for w in words if w not in stop_words]
            if filtered:
                word_counts = Counter(filtered).most_common(10)
                labels, values = zip(*word_counts)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(labels[::-1], values[::-1], color="steelblue")
                ax.set_xlabel("Frequency")
                ax.set_title("Top Words")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("### 🧹 Preprocessed Text")
            tokens = word_tokenize(text_clean.lower())
            clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
            st.code(" | ".join(clean_tokens))

        # ==========================
        # 5.4 MODEL DETAILS
        # ==========================
        with st.expander("ℹ️ Classifier details and training data"):
            st.markdown(
                "**Classifier:** TF‑IDF (1‑2 grams, 500 features) + Logistic Regression."
            )
            st.markdown(f"**Test accuracy on internal examples:** `{test_accuracy:.2f}`")
            st.markdown(
                "The model is trained on a small labelled dataset of aviation incidents "
                "for this course prototype."
            )

            st.markdown("**Training examples (category → sample text):**")
            for ex in get_training_data():
                em = CATEGORY_EMOJI.get(ex.category, "•")
                st.markdown(f"- {em} **{ex.category}** — {ex.text}")