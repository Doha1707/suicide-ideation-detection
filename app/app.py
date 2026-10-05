"""
Suicide Ideation Detection -- Research Prototype Interface (Streamlit)

IMPORTANT: This is a research prototype, NOT a diagnostic or clinical tool.
"""

import os
import re

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIG
# ============================================================
torch.manual_seed(0)
torch.set_num_threads(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "final_model_phi-2")
MAX_LEN = 128
DECISION_THRESHOLD = 0.50  

# ============================================================
# MODEL LOADING (cached: loads once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}. Put the folder "
            f"'final_model_phi-2' next to app.py."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, low_cpu_mem_usage=True, dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if use_gpu:
        model.to("cuda")
    return tokenizer, model


def classify_text(text, tokenizer, model):
    """Returns (probability_of_suicide_class, label)."""
    inputs = tokenizer(
        text, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)
        risk_prob = probs[0, 1].item()
    label = "Suicide" if risk_prob >= DECISION_THRESHOLD else "Non-suicide"
    return risk_prob, label


def render_single_result(risk_prob, label):
    if label == "Suicide":
        st.error(f"**Result: {label}**")
    else:
        st.success(f"**Result: {label}**")
    st.caption(f"Model confidence (probability of the 'Suicide' class): {risk_prob:.1%}")


def split_posts(raw_text, mode):
    if mode == "One post per line":
        return [line.strip() for line in raw_text.split("\n") if line.strip()]
    blocks = re.split(r"\n\s*\n", raw_text)
    return [b.strip().replace("\n", " ") for b in blocks if b.strip()]


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Suicide Ideation Detection -- Research Prototype")

st.title("Suicide Ideation Detection")

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("The model could not be loaded.")
    st.exception(e)
    st.stop()

tab_text, tab_batch = st.tabs(["Analyze text", "Analyze multiple posts"])

# ---------------- Single text ----------------
with tab_text:
    st.subheader("Enter a post or message")
    user_text = st.text_area(
        "Text to analyze", height=180, placeholder="Paste a post or message here..."
    )
    if st.button("Analyze", type="primary", key="single_analyze"):
        if not user_text.strip():
            st.error("Please enter some text before analyzing.")
        else:
            risk_prob, label = classify_text(user_text, tokenizer, model)
            render_single_result(risk_prob, label)

# ---------------- Multiple posts ----------------
with tab_batch:
    st.subheader("Analyze several posts at once")
    mode = st.radio(
        "How are your posts separated?",
        ["One post per line", "Separated by a blank line"],
        horizontal=True,
    )
    batch_text = st.text_area("Posts to analyze", height=220, key="batch_input")

    if st.button("Analyze all", type="primary", key="batch_analyze"):
        posts = split_posts(batch_text, mode)
        if not posts:
            st.error("Please enter at least one post before analyzing.")
        else:
            results = []
            progress = st.progress(0.0)
            for i, post in enumerate(posts):
                risk_prob, label = classify_text(post, tokenizer, model)
                results.append(
                    {
                        "#": i + 1,
                        "Post": post[:100] + ("..." if len(post) > 100 else ""),
                        "Result": label,
                        "Confidence": f"{risk_prob:.1%}",
                    }
                )
                progress.progress((i + 1) / len(posts))
            progress.empty()

            n_total = len(results)
            n_flagged = sum(1 for r in results if r["Result"] == "Suicide")
            pct_flagged = 100 * n_flagged / n_total

            st.subheader("Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Posts analyzed", n_total)
            c2.metric("Classified 'Suicide'", n_flagged)
            c3.metric("Share of 'Suicide' posts", f"{pct_flagged:.1f}%")

            st.subheader("Details per post")
            st.dataframe(results)

st.divider()
st.caption(
    "Model selected via a controlled comparison of six open-source LLMs on the "
    "Kaggle SuicideWatch dataset."
)