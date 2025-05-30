import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import pipeline
import wikipedia, re, pytesseract
from deep_translator import GoogleTranslator
from langdetect import detect
import re

# ---------------------------------------------------------
# 0️⃣  Load lightweight models just once (Streamlit caching)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    vision = tf.keras.applications.MobileNetV2(weights="imagenet")
    text_gen = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
        max_new_tokens=128,
        )

    return vision, text_gen

vision_model, fact_llm = load_models()

# ---------------------------------------------------------
# 1️⃣ Utility helpers
# ---------------------------------------------------------

def preprocess(img: Image.Image):
    """Resize → array → preprocess for MobileNetV2."""
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def top_pred(pred):
    return tf.keras.applications.mobilenet_v2.decode_predictions(pred, 1)[0][0]


def wiki_summary(label: str, sentences: int = 2) -> str:
    """Get a short Wikipedia summary if available."""
    try:
        return wikipedia.summary(label.split(",")[0], sentences=sentences, auto_suggest=False)
    except Exception:
        return "(No Wikipedia summary available.)"


def generate_facts(label: str, context: str) -> str:
    """Ask Flan‑T5 for 5 fun facts about the label, returning bullet points.
    Explicitly instruct the model NOT to repeat the background info."""

    prompt = (
        f"Write five short, surprising fun facts about {label}.\n"
        f"Do NOT repeat the following information; instead, provide new and interesting facts.\n"
        f"Use bullet points starting with • .\n\n"
        f"Background information (do not repeat):\n{context}\n\nFun facts:\n"
    )

    raw = fact_llm(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )[0]["generated_text"]

    # Try to find the bullet points start
    start = raw.find("•")
    if start == -1:
        # fallback: no bullets found, just take all after "Fun facts:"
        start = raw.lower().find("fun facts:")
        if start != -1:
            start += len("fun facts:")
            facts = raw[start:].strip()
        else:
            facts = raw.strip()
    else:
        facts = raw[start:].strip()

    # Normalize bullets formatting (replace any leading dash or number with bullet)
    facts = re.sub(r"^\s*[\-\d\.\•]+\s*", "• ", facts, flags=re.MULTILINE)

    # Extract only the first 5 bullet points
    bullets = re.findall(r"•\s.*", facts)
    if bullets:
        facts = "\n".join(bullets[:5])

    return facts


def detect_and_translate(text: str, target: str = "en") -> str:
    """Auto‑detect source language and translate with GoogleTranslator."""
    try:
        src = detect(text)
    except Exception:
        src = "auto"
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return "(Translation failed.)"

# ---------------------------------------------------------
# 2️⃣ Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Image → Facts & Info", page_icon="🖼️")
st.title("🖼️ Upload an Image → Get Instant Facts & Info")

uploaded = st.file_uploader("Upload a JPG / PNG image", ["jpg", "jpeg", "png"])
if not uploaded:
    st.info("⬆️  Choose an image to begin")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, use_container_width=True)

# ---------------------------------------------------------
# 3️⃣ Classification
# ---------------------------------------------------------
with st.spinner("Classifying with MobileNetV2 …"):
    pred = vision_model.predict(preprocess(img))
    _, label_raw, prob = top_pred(pred)
    prob *= 100
st.success(f"🔍 **Prediction:** {label_raw}  ({prob:.1f}% confidence)")

# ---------------------------------------------------------
# 4️⃣ Wikipedia summary & fun facts (LLM)
# ---------------------------------------------------------
summary = wiki_summary(label_raw)
with st.expander("ℹ️  Wikipedia summary"):
    st.write(summary)

# Only generate facts once unless the prediction label changes
if "cached_label" not in st.session_state or st.session_state.cached_label != label_raw:
    with st.spinner("Generating fun facts …"):
        facts = generate_facts(label_raw, summary)
        st.session_state.cached_label = label_raw
        st.session_state.cached_facts = facts

if st.session_state.get("cached_facts"):
    st.subheader("🎉 Fun Facts")

    # Convert bullet points to markdown list
    facts_lines = st.session_state.cached_facts.split("• ")
    formatted_facts = "\n".join([f"- {line.strip()}" for line in facts_lines if line.strip()])
    st.markdown(formatted_facts)


# ---------------------------------------------------------
# 5️⃣ OCR branch (always attempt)
# ---------------------------------------------------------
with st.spinner("Scanning image for text …"):
    extracted_text = pytesseract.image_to_string(img).strip()

if extracted_text:
    st.subheader("📝 Detected Text (raw)")
    st.code(extracted_text, language="text")

    # English version
    if detect_and_translate(extracted_text, "en").strip() != extracted_text.strip():
        st.markdown("**➡️ English translation:**")
        st.code(detect_and_translate(extracted_text, "en"))

    # Language selection dropdown
    lang_options = {
        "English": "en",
        "Hindi (हिंदी)": "hi",
        "Japanese (日本語)": "ja",
        "Spanish (Español)": "es",
        "French (Français)": "fr",
        "German (Deutsch)": "de",
        "Chinese (中文)": "zh-CN",
        "Auto Detect (no translation)": "auto",
    }

    selected_lang = st.selectbox("Select translation language:", list(lang_options.keys()), index=0)
    target_lang_code = lang_options[selected_lang]

    # Translate only if not 'auto' (which means no translation)
    if target_lang_code != "auto":
        translated_text = detect_and_translate(extracted_text, target_lang_code)
        st.markdown(f"**➡️ Translation to {selected_lang}:**")
        st.code(translated_text)
    else:
        st.info("Auto Detect selected — showing original text only.")

# ---------------------------------------------------------
# 6️⃣ Closing note
# ---------------------------------------------------------
st.caption("Built with TensorFlow, HuggingFace, and Wikipedia ✨")
