import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import pipeline
import wikipedia, re
from deep_translator import GoogleTranslator
from langdetect import detect
import easyocr                      # ⬅️  NEW

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
    ocr_reader = easyocr.Reader(["en"], gpu=False)   # cache EasyOCR
    return vision, text_gen, ocr_reader

vision_model, fact_llm, ocr_reader = load_models()

# ---------------------------------------------------------
# 1️⃣ Utility helpers
# ---------------------------------------------------------

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

def top_pred(pred):
    return tf.keras.applications.mobilenet_v2.decode_predictions(pred, 1)[0][0]

def wiki_summary(label: str, sentences: int = 2) -> str:
    try:
        return wikipedia.summary(label.split(",")[0], sentences=sentences, auto_suggest=False)
    except Exception:
        return "(No Wikipedia summary available.)"

def generate_facts(label: str, context: str) -> str:
    prompt = (
        f"Write five short, surprising fun facts about {label}.\n"
        f"Do NOT repeat the following information; instead, provide new and interesting facts.\n"
        f"Use bullet points starting with • .\n\n"
        f"Background information (do not repeat):\n{context}\n\nFun facts:\n"
    )
    raw = fact_llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]

    # find first bullet
    start = raw.find("•")
    if start == -1:
        start = raw.lower().find("fun facts:")
        facts = raw[start + len("fun facts:") :].strip() if start != -1 else raw.strip()
    else:
        facts = raw[start:].strip()

    facts = re.sub(r"^\s*[\-\d\.\•]+\s*", "• ", facts, flags=re.MULTILINE)
    bullets = re.findall(r"•\s.*", facts)
    return "\n".join(bullets[:5]) if bullets else facts

def detect_and_translate(text: str, target: str = "en") -> str:
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

if "cached_label" not in st.session_state or st.session_state.cached_label != label_raw:
    with st.spinner("Generating fun facts …"):
        facts = generate_facts(label_raw, summary)
        st.session_state.cached_label = label_raw
        st.session_state.cached_facts = facts

if st.session_state.get("cached_facts"):
    st.subheader("🎉 Fun Facts")
    facts_lines = st.session_state.cached_facts.split("• ")
    st.markdown("\n".join(f"- {line.strip()}" for line in facts_lines if line.strip()))

# ---------------------------------------------------------
# 5️⃣ OCR branch (EasyOCR)
# ---------------------------------------------------------
with st.spinner("Scanning image for text …"):
    # EasyOCR expects numpy array
    results = ocr_reader.readtext(np.array(img))
    extracted_text = "\n".join([res[1] for res in results]).strip()

if extracted_text:
    st.subheader("📝 Detected Text (raw)")
    st.code(extracted_text, language="text")

    if detect_and_translate(extracted_text, "en").strip() != extracted_text.strip():
        st.markdown("**➡️ English translation:**")
        st.code(detect_and_translate(extracted_text, "en"))

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
    if target_lang_code != "auto":
        translated_text = detect_and_translate(extracted_text, target_lang_code)
        st.markdown(f"**➡️ Translation to {selected_lang}:**")
        st.code(translated_text)
    else:
        st.info("Auto Detect selected — showing original text only.")

# ---------------------------------------------------------
# 6️⃣ Closing note
# ---------------------------------------------------------
st.caption("Built with TensorFlow, HuggingFace, EasyOCR, and Wikipedia ✨")
