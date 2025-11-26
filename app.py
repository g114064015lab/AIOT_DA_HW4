import json
import re
from functools import lru_cache

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 450
DEFAULT_TEMPERATURE = 0.2


@st.cache_resource(show_spinner=False)
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )


@lru_cache(maxsize=4)
def system_prompt():
    return (
        "You are an analyst for IMDB movie reviews. "
        "Return a single JSON object answering all tasks. "
        "Use concise sentences; keep lists under 4 items. "
        "Keys required: sentiment_label, topic, summary, sentiment_score_10, "
        "key_sentences, keywords, rating_pred_10, audience_suitability. "
        "sentiment_label must be one of "
        "[Positive, Neutral, Negative, Touched, Angry, Disappointed, Surprised]. "
        "topic must be one of [Plot, Acting, Directing, Visual Effects, Music, Pacing, Other]. "
        "summary must preserve the review's sentiment. "
        "sentiment_score_10 is 0-10 reflecting emotion intensity. "
        "key_sentences: short list of pivotal sentences. "
        "keywords: short list of representative terms. "
        "rating_pred_10: predicted rating 0-10 from the review. "
        "audience_suitability: one sentence describing who will like or dislike the film."
    )


def build_prompt(review: str, temperature: float):
    tokenizer = load_generator().tokenizer
    messages = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": (
                "Analyze the following IMDB review and respond ONLY with JSON. "
                "Do not add explanations.\n\n"
                f"Review:\n{review.strip()}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_json_block(text: str):
    cleaned = text.strip()
    fenced = re.search(r"```json(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    braces = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if braces:
        cleaned = braces.group(0)
    cleaned = cleaned.replace("\u3000", " ").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(cleaned.replace("'", '"'))
    except Exception:
        return None


def analyze_review(review: str, temperature: float):
    generator = load_generator()
    prompt = build_prompt(review, temperature)
    outputs = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=generator.tokenizer.eos_token_id,
    )
    raw = outputs[0]["generated_text"][len(prompt) :]
    parsed = extract_json_block(raw)
    return parsed, raw


def render_results(parsed: dict, raw: str):
    st.subheader("分析結果")
    col1, col2 = st.columns(2)

    col1.metric("情緒/心情 (7類)", parsed.get("sentiment_label", "N/A"))
    col1.metric("主題", parsed.get("topic", "N/A"))
    col1.metric(
        "情緒強度 /10", f"{parsed.get('sentiment_score_10', 'N/A')}"
    )
    col1.metric(
        "可能評分 /10", f"{parsed.get('rating_pred_10', 'N/A')}"
    )

    col2.write("**摘要（保留情緒）**")
    col2.write(parsed.get("summary", ""))

    col2.write("**觀眾適配**")
    col2.write(parsed.get("audience_suitability", ""))

    st.write("---")
    st.write("**關鍵句**")
    key_sentences = parsed.get("key_sentences") or []
    for idx, sentence in enumerate(key_sentences, 1):
        st.write(f"{idx}. {sentence}")

    st.write("**關鍵字**")
    keywords = parsed.get("keywords") or []
    st.write(", ".join(keywords))

    with st.expander("查看原始模型輸出"):
        st.code(raw)


def main():
    st.set_page_config(
        page_title="IMDB Review 7合1 情緒/主題/摘要分析",
        page_icon="🎬",
        layout="wide",
    )
    st.title("🎬 IMDB 影評 7 合 1 智能分析")
    st.caption(
        "多類情緒、主題、情緒保留摘要、強度、關鍵句/詞、評分與觀眾適配，一次完成。"
    )

    example = (
        "The film's pacing is uneven, but the acting is heartfelt. "
        "I laughed a few times, yet the ending felt rushed and predictable. "
        "Overall, it's a decent weekend watch, nothing mind-blowing."
    )

    with st.sidebar:
        st.header("推理設定")
        temperature = st.slider(
            "溫度 (較低=穩定, 較高=多樣)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
        )
        st.markdown(
            "模型：`Qwen/Qwen2.5-1.5B-Instruct`（本地推理，僅 JSON 回傳）"
        )

    review = st.text_area(
        "貼上 IMDB 影評文字",
        value=example,
        height=180,
        placeholder="輸入英文或中英混合影評，按下分析開始推理。",
    )

    if st.button("開始分析", type="primary"):
        if not review.strip():
            st.warning("請先輸入影評文字。")
        else:
            with st.spinner("模型推理中，請稍候..."):
                parsed, raw = analyze_review(review, temperature)
            if parsed:
                render_results(parsed, raw)
            else:
                st.error("無法解析模型輸出，請再試一次或調整溫度。")
                with st.expander("原始輸出"):
                    st.code(raw)


if __name__ == "__main__":
    main()
