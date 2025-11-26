import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
HEURISTICS_FILE = MODEL_DIR / os.getenv("IMDB_MODEL_FILE", "heuristics.json")
DEFAULT_TEMPERATURE = 0.2
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "to",
    "is",
    "it",
    "in",
    "that",
    "for",
    "this",
    "on",
    "with",
    "as",
    "was",
    "but",
    "are",
    "be",
    "at",
    "by",
    "from",
    "or",
    "so",
}
WORD_RE = re.compile(r"[A-Za-z']+")


@st.cache_resource(show_spinner=False)
def load_local_model():
    if not HEURISTICS_FILE.exists():
        raise FileNotFoundError(
            f"找不到模型設定檔：{HEURISTICS_FILE}. 請確認 model 資料夾已同步。"
        )
    with open(HEURISTICS_FILE, "r", encoding="utf-8") as fp:
        heuristics = json.load(fp)
    analyzer = SentimentIntensityAnalyzer()
    return heuristics, analyzer


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def detect_sentiment_label(
    text: str, compound: float, keywords: Dict[str, List[str]]
) -> str:
    lowered = text.lower()
    for label, cues in keywords.items():
        if any(cue in lowered for cue in cues):
            return label
    if compound >= 0.35:
        return "Positive"
    if compound <= -0.35:
        return "Negative"
    if abs(compound) <= 0.1:
        return "Neutral"
    return "Disappointed" if compound < 0 else "Touched"


def detect_topic(text: str, topic_keywords: Dict[str, List[str]]) -> str:
    lowered = text.lower()
    scores = {}
    for topic, cues in topic_keywords.items():
        scores[topic] = sum(lowered.count(cue) for cue in cues)
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "Other"


def select_keywords(tokens: List[str], top_n: int = 6) -> List[str]:
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def score_sentence(
    sentence: str, analyzer: SentimentIntensityAnalyzer, diversity: float
) -> float:
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    sentiment = abs(analyzer.polarity_scores(sentence)["compound"])
    return len(tokens) + sentiment * (5 + diversity * 5)


def summarize(
    sentences: List[str], analyzer: SentimentIntensityAnalyzer, diversity: float
) -> List[str]:
    scored = sorted(
        ((score_sentence(s, analyzer, diversity), s) for s in sentences),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [s for _, s in scored[: min(3, len(scored))]]


def calc_intensity(compound: float) -> float:
    return round(min(1.0, abs(compound)) * 10, 2)


def rating_from_sentiment(compound: float) -> float:
    rating = ((compound + 1) / 2) * 10
    return round(max(0, min(10, rating)), 1)


def audience_suggestion(text: str, profiles: List[Dict[str, List[str]]]) -> str:
    lowered = text.lower()
    best_label = "適合尋找劇情/角色深度的觀眾，避免期待爆米花娛樂的人。"
    best_score = 0
    for profile in profiles:
        score = sum(lowered.count(keyword) for keyword in profile["keywords"])
        if score > best_score:
            best_score = score
            best_label = profile["label"]
    return best_label


def analyze_review(review: str, diversity: float):
    heuristics, analyzer = load_local_model()
    text = review.strip()
    compound = analyzer.polarity_scores(text)["compound"]
    sentiment_label = detect_sentiment_label(
        text, compound, heuristics["sentiment_keywords"]
    )
    topic = detect_topic(text, heuristics["topic_keywords"])
    sentences = split_sentences(text)
    highlight_sentences = summarize(sentences, analyzer, diversity)
    summary = " ".join(highlight_sentences) if highlight_sentences else text
    tokens = tokenize(text)
    keywords = select_keywords(tokens)
    return {
        "sentiment_label": sentiment_label,
        "topic": topic,
        "summary": summary,
        "sentiment_score_10": calc_intensity(compound),
        "key_sentences": highlight_sentences,
        "keywords": keywords,
        "rating_pred_10": rating_from_sentiment(compound),
        "audience_suitability": audience_suggestion(
            text, heuristics["audience_profiles"]
        ),
    }


def render_results(parsed: dict):
    st.subheader("分析結果")
    col1, col2 = st.columns(2)

    col1.metric("情緒/心情 (7類)", parsed.get("sentiment_label", "N/A"))
    col1.metric("主題", parsed.get("topic", "N/A"))
    col1.metric("情緒強度 /10", f"{parsed.get('sentiment_score_10', 'N/A')}")
    col1.metric("可能評分 /10", f"{parsed.get('rating_pred_10', 'N/A')}")

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
            "調整情緒靈敏度（僅影響關鍵句排序）",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="越高表示更偏好情緒波動大的句子，0 則偏好關鍵資訊。",
        )
        st.markdown(
            f"模型：`{HEURISTICS_FILE}`（本地規則/字典，無需雲端下載）"
        )

    review = st.text_area(
        "貼上 IMDB 影評文字",
        value=example,
        height=180,
        placeholder="輸入英文或中英混合影評，按下分析開始。",
    )

    if st.button("開始分析", type="primary"):
        if not review.strip():
            st.warning("請先輸入影評文字。")
        else:
            with st.spinner("本地模型分析中，請稍候..."):
                parsed = analyze_review(review, temperature)
            render_results(parsed)


if __name__ == "__main__":
    main()
