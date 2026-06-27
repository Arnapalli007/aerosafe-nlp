import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import re
import spacy
import numpy as np
import time
import io
import csv
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# NLTK downloads
# --------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# --------------------------------------------------------------------
# SAMPLE TEXT
# --------------------------------------------------------------------
SAMPLE_TEXT = """During approach to runway 28L at Frankfurt Airport (EDDF) on 14 March 2024,
the crew of flight DLH452 encountered unexpected windshear at approximately 800 feet AGL.
The captain initiated a go-around after the airspeed dropped 18 knots below target speed.
ATC was notified immediately and a new approach clearance was issued. Post-flight inspection
by Lufthansa maintenance revealed no structural damage. The first officer filed an ASRS report
citing crew fatigue following a 12-hour duty period as a contributing factor. The Boeing 737-800
landed safely on the second approach at 14:32 UTC."""

# --------------------------------------------------------------------
# CATEGORY KEYWORDS for text windowing
# --------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Weather": ["turbulence", "windshear", "thunderstorm", "lightning", "fog", "icing",
                "microburst", "visibility", "precipitation", "storm", "wind", "weather"],
    "Bird Strike": ["bird", "birds", "wildlife", "strike", "ingested", "flock",
                    "feathers", "avian", "raptor", "gull", "wildlife"],
    "Equipment Failure": ["failure", "malfunction", "hydraulic", "engine", "fuel cutoff",
                          "warning", "autopilot", "gear", "flap", "system failure",
                          "cutoff", "shutdown", "rat", "ram air turbine", "fadec",
                          "fuel control", "switch", "n1", "n2", "egt", "relight"],
    "ATC Issue": ["atc", "clearance", "controller", "tcas", "frequency", "radar",
                  "runway incursion", "readback", "squawk", "separation", "instruction"],
    "Human Factors": ["fatigue", "crew", "error", "disorientation", "workload",
                      "checklist", "crm", "distraction", "pilot", "rushed",
                      "duty hours", "rest period", "spatial", "violated", "missed"],
    "Maintenance": ["maintenance", "inspection", "technician", "torque", "servicing",
                    "repair", "overhaul", "fitting", "component", "mechanic", "mel",
                    "airworthiness", "defect", "amc", "engineer"],
}


def extract_relevant_window(text: str, max_sentences: int = 40) -> str:
    """
    For long documents: score each sentence by aviation safety keyword density
    and return the top-scoring sentences for classification.
    This prevents long multi-topic PDFs from confusing the classifier.
    """
    all_keywords = set()
    for kws in CATEGORY_KEYWORDS.values():
        all_keywords.update(kws)

    sentences = sent_tokenize(text)

    # If short document, use as-is
    if len(sentences) <= max_sentences:
        return text

    # Score each sentence by keyword density
    scored = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        score = sum(1 for kw in all_keywords if kw in sent_lower)
        scored.append((score, i, sent))

    # Take top sentences, preserve original order
    top = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join(s for _, _, s in top_sorted)


# --------------------------------------------------------------------
# 1. CLASSIFIER DATA & MODEL
# --------------------------------------------------------------------
@dataclass
class IncidentExample:
    text: str
    category: str


def get_training_data() -> List[IncidentExample]:
    examples = [
        # Weather (6)
        IncidentExample("The aircraft encountered severe turbulence during cruise at FL350. Seat belt signs were activated.", "Weather"),
        IncidentExample("Heavy thunderstorms forced a diversion to an alternate airport. Lightning observed near approach path.", "Weather"),
        IncidentExample("Severe windshear encountered during final approach. Go-around initiated at decision height.", "Weather"),
        IncidentExample("Microburst activity reported by ATIS. Wind changed from strong headwind to strong tailwind within seconds.", "Weather"),
        IncidentExample("Icing conditions caused pitot tube blockage during descent. Unreliable airspeed indications noted.", "Weather"),
        IncidentExample("Low visibility fog at destination required diversion. CAT III autoland performed at alternate.", "Weather"),

        # Bird Strike (6)
        IncidentExample("Bird strike occurred during takeoff roll. Left engine ingested a large bird. Rejected takeoff performed.", "Bird Strike"),
        IncidentExample("Flock of birds seen on runway during approach. Crew executed immediate go-around.", "Bird Strike"),
        IncidentExample("Multiple gulls struck nose and windshield during rotation. Post-flight inspection revealed minor damage.", "Bird Strike"),
        IncidentExample("Crew reported bird impact on right wing during climb through 3000 feet. No abnormal indications.", "Bird Strike"),
        IncidentExample("Large raptor ingested into engine number two on departure. Engine shutdown procedure completed.", "Bird Strike"),
        IncidentExample("Bird remains found on fuselage after landing. Wildlife strike report filed with airport authority.", "Bird Strike"),

        # Equipment Failure (12 — expanded)
        IncidentExample("Hydraulic system failure indicated during climb. Emergency checklist completed. Aircraft landed safely.", "Equipment Failure"),
        IncidentExample("Engine oil pressure warning light illuminated shortly after takeoff. Precautionary landing at nearest airport.", "Equipment Failure"),
        IncidentExample("Gear retraction failed after takeoff. Crew performed abnormal checklist and extended gear manually.", "Equipment Failure"),
        IncidentExample("Autopilot disconnected unexpectedly during approach. Manual flying required in low visibility conditions.", "Equipment Failure"),
        IncidentExample("Oxygen system failed during high altitude cruise. Crew initiated emergency descent and diverted.", "Equipment Failure"),
        IncidentExample("Flap asymmetry warning activated on final approach. Go-around initiated and flaps retracted.", "Equipment Failure"),
        IncidentExample("Radio altimeter gave spurious readings during ILS approach. Crew reverted to barometric altitude.", "Equipment Failure"),
        IncidentExample("Both fuel control switches transitioned from RUN to CUTOFF after takeoff. Engines N1 and N2 began to decrease. Ram Air Turbine deployed automatically.", "Equipment Failure"),
        IncidentExample("Engine fuel cutoff switch inadvertently moved to CUTOFF position during initial climb. FADEC attempted automatic relight sequence.", "Equipment Failure"),
        IncidentExample("Ram Air Turbine deployed after dual engine power loss. APU auto start logic activated. Crew declared MAYDAY.", "Equipment Failure"),
        IncidentExample("Fuel control switch locking feature disengaged during takeoff roll. Both engines lost thrust. Emergency descent initiated.", "Equipment Failure"),
        IncidentExample("Engine N2 core speed dropped below minimum idle after fuel cutoff. FADEC relight attempt unsuccessful on engine two.", "Equipment Failure"),

        # ATC Issue (6)
        IncidentExample("First officer read back incorrect altitude clearance. Crew did not cross-check. TCAS RA was issued.", "ATC Issue"),
        IncidentExample("Controller issued conflicting clearances to two aircraft on parallel approach. TCAS activated.", "ATC Issue"),
        IncidentExample("ATC cleared aircraft to wrong runway. Crew queried the clearance and potential conflict was avoided.", "ATC Issue"),
        IncidentExample("Tower frequency congestion resulted in missed landing clearance. Readback error not caught.", "ATC Issue"),
        IncidentExample("Incorrect squawk code assigned by radar controller. Aircraft temporarily lost on radar display.", "ATC Issue"),
        IncidentExample("Runway incursion due to unclear taxi instructions. Ground controller issued hold short instruction late.", "ATC Issue"),

        # Human Factors (12 — expanded)
        IncidentExample("Fatigue due to extended duty hours led to missed callout during approach briefing.", "Human Factors"),
        IncidentExample("Captain experienced spatial disorientation in IMC during non-precision approach.", "Human Factors"),
        IncidentExample("Crew resource management breakdown during high-workload approach in icing conditions.", "Human Factors"),
        IncidentExample("Pilot rushed pre-flight checklist after late arrival, resulting in flap configuration error on takeoff.", "Human Factors"),
        IncidentExample("First officer failed to monitor altitude during high workload climb phase due to distraction.", "Human Factors"),
        IncidentExample("Sterile cockpit rule violated below 10000 feet. Non-essential conversation distracted crew.", "Human Factors"),
        IncidentExample("Co-pilot inadvertently moved fuel cutoff switch to CUTOFF position during initial climb phase after takeoff.", "Human Factors"),
        IncidentExample("Pilot error caused fuel control switch to be moved to cutoff. Other pilot asked why did he cutoff. Crew confusion during emergency.", "Human Factors"),
        IncidentExample("Crew member accidentally activated fuel shutoff during climb. Spatial disorientation suspected. Emergency declared.", "Human Factors"),
        IncidentExample("Pilot monitoring inadvertently reached for wrong switch. Fuel control lever moved to cutoff position during high workload.", "Human Factors"),
        IncidentExample("Inadequate crew coordination led to wrong switch selection during initial climb. Both engines affected by pilot error.", "Human Factors"),
        IncidentExample("First officer with limited experience on type made switch selection error during takeoff climb phase causing dual engine event.", "Human Factors"),

        # Maintenance (8 — slightly expanded)
        IncidentExample("Maintenance overlooked cracked brake assembly during pre-flight inspection, discovered after landing.", "Maintenance"),
        IncidentExample("Hydraulic fitting not torqued to specification during maintenance visit, causing slow leak.", "Maintenance"),
        IncidentExample("Incorrect tire pressure recorded by maintenance personnel, requiring replacement before departure.", "Maintenance"),
        IncidentExample("Oil cap not secured properly after engine servicing, resulting in oil streaks observed on nacelle.", "Maintenance"),
        IncidentExample("Wrong engine oil type used during line maintenance. Engine oil consumption increased abnormally.", "Maintenance"),
        IncidentExample("Avionics panel not secured after scheduled maintenance. Panel vibrated loose during cruise.", "Maintenance"),
        IncidentExample("MEL item for fuel control switch locking feature was not inspected during scheduled maintenance visit.", "Maintenance"),
        IncidentExample("Throttle control module replaced but fuel control switch locking feature not verified during post-maintenance check.", "Maintenance"),
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

CATEGORY_COLORS = {
    "Weather": "#4A90D9",
    "Bird Strike": "#7BB241",
    "Equipment Failure": "#E08B30",
    "ATC Issue": "#9B59B6",
    "Human Factors": "#E74C3C",
    "Maintenance": "#1ABC9C",
}


@st.cache_resource(show_spinner=True)
def train_classifier() -> Tuple[TfidfVectorizer, LogisticRegression, float]:
    data = get_training_data()
    texts = [ex.text for ex in data]
    labels = [ex.category for ex in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train_tfidf, y_train)
    accuracy = clf.score(X_test_tfidf, y_test)

    return vectorizer, clf, accuracy


def extract_key_sentence(text: str, category: str) -> str:
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
    if not sentences:
        return text.strip()[:200]

    hints = set(CATEGORY_KEYWORDS.get(category, []))
    best_score = -1
    best_sentence = sentences[0]
    for s in sentences:
        words = set(s.lower().split())
        score = len(words & hints) * 2 + len(s.split()) * 0.05
        if score > best_score:
            best_score = score
            best_sentence = s
    return best_sentence


# --------------------------------------------------------------------
# 2. SPACY + VADER
# --------------------------------------------------------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


nlp = load_spacy_model()
vader = load_vader()


# --------------------------------------------------------------------
# 3. CSV EXPORT
# --------------------------------------------------------------------
def build_csv_export(results: dict) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Section", "Key", "Value"])
    writer.writerow(["Classification", "Predicted Category", results["category"]])
    writer.writerow(["Classification", "Confidence (%)", f"{results['confidence']:.1f}"])
    writer.writerow(["Sentiment", "Score", f"{results['sentiment_score']:.3f}"])
    writer.writerow(["Sentiment", "Label", results["sentiment_label"]])
    writer.writerow(["Stats", "Characters", results["chars"]])
    writer.writerow(["Stats", "Words", results["words"]])
    writer.writerow(["Stats", "Sentences", results["sentences"]])
    writer.writerow(["Stats", "Latency (ms)", f"{results['latency_ms']:.0f}"])
    writer.writerow([])
    writer.writerow(["Keywords", "Rank", "Phrase"])
    for i, kw in enumerate(results["keywords"], 1):
        writer.writerow(["Keywords", i, kw])
    writer.writerow([])
    writer.writerow(["Named Entities", "Text", "Label"])
    for ent_text, ent_label in results["entities"]:
        writer.writerow(["NER", ent_text, ent_label])
    writer.writerow([])
    writer.writerow(["Category Probabilities", "Category", "Probability (%)"])
    for cls, p in results["all_probs"]:
        writer.writerow(["Probability", cls, f"{p*100:.1f}"])
    return output.getvalue().encode("utf-8")


# --------------------------------------------------------------------
# 4. PAGE CONFIG & SIDEBAR
# --------------------------------------------------------------------
st.set_page_config(page_title="AeroSafe NLP", page_icon="✈️", layout="wide")

with st.sidebar:
    st.markdown(
        '<div style="font-size:56px;text-align:center;margin-bottom:4px;">✈️</div>',
        unsafe_allow_html=True,
    )
    st.title("AeroSafe NLP")
    st.markdown("---")
    st.markdown("**NLP Features:**")
    st.markdown("✅ Incident Category Classifier")
    st.markdown("✅ Sentiment Analysis (VADER)")
    st.markdown("✅ Named Entity Recognition")
    st.markdown("✅ Keyword Extraction (RAKE)")
    st.markdown("✅ Word Frequency Chart")
    st.markdown("✅ Text Preprocessing")
    st.markdown("✅ File Upload (.txt / .csv / .pdf)")
    st.markdown("✅ CSV Export of Results")
    st.markdown("---")
    st.markdown("**Model:** TF-IDF + Logistic Regression")
    st.markdown("**Sentiment:** VADER (lexicon-based)")
    st.markdown("**NER:** spaCy en_core_web_sm")
    st.markdown("---")
    st.markdown("*Built by Gayatri Naidu Arnapalli*")
    st.markdown("*HFU IBS Bachelor Project 2026*")

st.title("✈️ AeroSafe NLP Prototype")
st.write("Analyze aviation safety reports using Natural Language Processing.")

# --------------------------------------------------------------------
# 5. INPUT SECTION
# --------------------------------------------------------------------
st.markdown("### 📂 Input Text")

if st.button("📋 Try Sample Report", help="Load a pre-written aviation incident example"):
    st.session_state["sample_text"] = SAMPLE_TEXT

input_method = st.radio(
    "Choose input method:",
    ["Type / Paste text", "Upload a .txt file", "Upload a .csv file", "Upload a .pdf file"],
    horizontal=True,
)

user_text = ""

if input_method == "Type / Paste text":
    default_val = st.session_state.get("sample_text", "")
    user_text = st.text_area(
        "📝 Enter aviation safety text to analyze:",
        value=default_val,
        height=150,
        placeholder="e.g. Boeing 737 reported a safety incident at London Heathrow on Monday...",
    )
    if default_val:
        st.info("💡 Sample report loaded — click Analyze to see results.")

elif input_method == "Upload a .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")
        st.success(f"✅ File loaded: **{uploaded_file.name}** ({len(user_text)} characters)")
        st.text_area("📄 Preview:", value=user_text[:500] + ("..." if len(user_text) > 500 else ""), height=120, disabled=True)

elif input_method == "Upload a .csv file":
    uploaded_file = st.file_uploader("Upload a .csv file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ CSV loaded: **{uploaded_file.name}** ({len(df)} rows)")
        text_column = st.selectbox("Select the column with report text:", df.columns)
        user_text = "\n".join(df[text_column].astype(str).tolist())
        st.text_area("📄 Preview:", value=user_text[:500] + ("..." if len(user_text) > 500 else ""), height=120, disabled=True)

elif input_method == "Upload a .pdf file":
    uploaded_file = st.file_uploader("Upload a .pdf file", type=["pdf"])
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        text_chunks = [page.extract_text() or "" for page in reader.pages]
        user_text = "\n".join(text_chunks)
        st.success(f"✅ PDF loaded: **{uploaded_file.name}** ({len(reader.pages)} pages)")
        st.text_area("📄 Preview:", value=user_text[:500] + ("..." if len(user_text) > 500 else ""), height=120, disabled=True)

with st.spinner("Loading classifier (first run only)…"):
    vectorizer, clf, test_accuracy = train_classifier()

# --------------------------------------------------------------------
# 6. ANALYZE
# --------------------------------------------------------------------
if st.button("🔍 Analyze", use_container_width=True, type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text or upload a file first.")
    else:
        text_clean = user_text.strip()
        t_start = time.time()

        # TEXT WINDOWING for long documents
        is_long_doc = len(sent_tokenize(text_clean)) > 40
        classification_text = extract_relevant_window(text_clean)

        # Classification
        X_input = vectorizer.transform([classification_text])
        proba = clf.predict_proba(X_input)[0]
        classes = clf.classes_
        predicted_index = int(np.argmax(proba))
        predicted_category = classes[predicted_index]
        confidence = proba[predicted_index] * 100

        # Sentiment (VADER)
        scores = vader.polarity_scores(text_clean)
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Keywords
        rake = Rake()
        rake.extract_keywords_from_text(text_clean)
        keywords = rake.get_ranked_phrases()[:5]

        # NER
        ner_text = text_clean[:5000]
        doc = nlp(ner_text)
        entity_icons = {
            "ORG": "🏢", "GPE": "🌍", "PERSON": "👤", "DATE": "📅",
            "TIME": "⏰", "LOC": "📍", "PRODUCT": "📦", "EVENT": "🎯",
            "FAC": "🛫", "QUANTITY": "📏", "CARDINAL": "🔢"
        }
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.text.strip().lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append((ent.text, ent.label_))

        # Stats
        stop_words = set(stopwords.words("english"))
        sentence_count = len([s for s in re.split(r'[.!?]', text_clean) if s.strip()])
        words_list = re.findall(r"\b[a-zA-Z]{3,}\b", text_clean.lower())
        filtered_words = [w for w in words_list if w not in stop_words]
        word_counts = Counter(filtered_words).most_common(10)

        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000

        key_sentence = extract_key_sentence(text_clean, predicted_category)

        results_for_export = {
            "category": predicted_category,
            "confidence": confidence,
            "sentiment_score": compound,
            "sentiment_label": sentiment_label,
            "chars": len(text_clean),
            "words": len(text_clean.split()),
            "sentences": sentence_count,
            "latency_ms": latency_ms,
            "keywords": keywords,
            "entities": entities,
            "all_probs": sorted(zip(classes, proba), key=lambda x: -x[1]),
        }

        # --- DISPLAY: CLASSIFICATION ---
        st.markdown("---")
        st.markdown("## 1️⃣ Incident Category Classifier")

        if is_long_doc:
            st.info("📄 Long document detected — classifier applied keyword-based text windowing for improved accuracy.")

        color = CATEGORY_COLORS.get(predicted_category, "#666")
        emoji = CATEGORY_EMOJI.get(predicted_category, "✈️")

        clf_left, clf_right = st.columns([1, 2])
        with clf_left:
            st.markdown(
                f'<div style="background:{color}22;border-left:4px solid {color};'
                f'padding:16px;border-radius:8px;">'
                f'<div style="font-size:2rem">{emoji}</div>'
                f'<div style="font-size:1.2rem;font-weight:bold;color:{color}">{predicted_category}</div>'
                f'<div style="font-size:0.9rem;color:#555">Confidence: <b>{confidence:.1f}%</b></div>'
                f'<div style="font-size:0.75rem;color:#888;margin-top:4px">⏱ {latency_ms:.0f} ms</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if confidence < 40:
                st.warning("⚠️ Low confidence — this report may span multiple incident types.")

        with clf_right:
            st.markdown("**Category probabilities:**")
            for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                em = CATEGORY_EMOJI.get(cls, "•")
                st.markdown(f"{em} **{cls}** — {p*100:.1f}%")
                st.progress(float(p))

        st.markdown("### 📝 Key Insight")
        st.info(f'"{key_sentence}"')

        # --- DISPLAY: NLP ANALYSIS ---
        st.markdown("---")
        st.markdown("## 2️⃣ Advanced NLP Analysis")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📊 Text Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Characters", len(text_clean))
            c2.metric("Words", len(text_clean.split()))
            c3.metric("Sentences", sentence_count)

            st.markdown("### 💬 Sentiment (VADER)")
            if compound >= 0.05:
                st.success(f"😊 Positive — compound score: **{compound:.3f}**")
            elif compound <= -0.05:
                st.error(f"😠 Negative — compound score: **{compound:.3f}**")
            else:
                st.info(f"😐 Neutral — compound score: **{compound:.3f}**")
            st.caption(f"pos={scores['pos']:.2f}  neu={scores['neu']:.2f}  neg={scores['neg']:.2f}")

            st.markdown("### 🔑 Keywords (RAKE)")
            if keywords:
                for kw in keywords:
                    st.write(f"🔹 {kw}")
            else:
                st.info("No keywords extracted. Try longer text.")

            st.markdown("### 🧠 Named Entities (spaCy)")
            if entities:
                for ent_text, ent_label in entities[:30]:
                    icon = entity_icons.get(ent_label, "🔷")
                    st.write(f"{icon} **{ent_text}** — `{ent_label}`")
                if len(entities) > 30:
                    st.caption(f"... and {len(entities) - 30} more entities")
            else:
                st.info("No named entities found.")

        with col_right:
            st.markdown("### 📈 Word Frequency Chart")
            if word_counts:
                labels_wf, values_wf = zip(*word_counts)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_wf = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels_wf)))
                ax.barh(labels_wf[::-1], values_wf[::-1], color=colors_wf[::-1])
                ax.set_xlabel("Frequency")
                ax.set_title("Top Words (stop words removed)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("### 🧹 Preprocessed Tokens")
            tokens = word_tokenize(text_clean.lower())
            clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
            st.code(" | ".join(clean_tokens[:40]) + (" ..." if len(clean_tokens) > 40 else ""))

        # --- CSV EXPORT ---
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        csv_bytes = build_csv_export(results_for_export)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_bytes,
            file_name="aerosafe_nlp_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # --- MODEL DETAILS ---
        with st.expander("ℹ️ Model Details & Training Data"):
            st.markdown("**Classifier:** TF-IDF (1–2 grams, 1000 features, sublinear TF) + Logistic Regression (C=1.0)")
            st.markdown(f"**Internal test accuracy:** `{test_accuracy:.2f}` (on held-out 20% of training data)")
            st.markdown("**Long document handling:** Keyword-based text windowing (top 40 sentences by aviation keyword density)")
            st.markdown("**Sentiment:** VADER — lexicon-based, calibrated for short informal texts")
            st.markdown("**NER:** spaCy `en_core_web_sm` — general English NER pipeline")
            st.markdown("---")
            st.markdown("**Training examples per category:**")
            from collections import Counter as C
            cat_counts = C(ex.category for ex in get_training_data())
            for cat, count in sorted(cat_counts.items()):
                em = CATEGORY_EMOJI.get(cat, "•")
                st.markdown(f"- {em} **{cat}**: {count} examples")