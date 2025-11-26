import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
HEURISTICS_FILE = MODEL_DIR / os.getenv("IMDB_MODEL_FILE", "heuristics.json")
DATASET_DIR = BASE_DIR / "IMDb影評"
DATASET_FILE = DATASET_DIR / "reviews.jsonl"
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


@st.cache_data(show_spinner=False)
def load_review_dataset() -> List[Dict[str, str]]:
    if not DATASET_FILE.exists():
        return []
    rows: List[Dict[str, str]] = []
    with open(DATASET_FILE, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def set_review_text(text: str):
    st.session_state["input_review"] = text


def set_random_review(rows: List[Dict[str, str]]):
    if rows:
        set_review_text(random.choice(rows)["review"])


def build_chart_dataframe(rows: List[Dict[str, str]], field: str):
    counter = Counter(row.get(field, "Unknown") for row in rows)
    if not counter:
        return None
    data = pd.DataFrame(
        {"label": list(counter.keys()), "count": list(counter.values())}
    ).sort_values("count", ascending=False)
    return data


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


def select_keywords(tokens: List[str], top_n: int) -> List[str]:
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


def audience_suggestion_from_rating(score: float) -> str:
    buckets = [
        (8.5, 11.0, "推薦尋找高品質敘事與演技的觀眾，幾乎可放心衝首輪。"),
        (7.0, 8.5, "適合該題材粉絲或想找精緻娛樂的觀眾，戲院或串流都值得。"),
        (5.0, 7.0, "建議輕鬆看片或串流觀賞，挑細節的觀眾可能覺得普通。"),
        (3.0, 5.0, "僅推薦鐵粉或想吐槽的觀眾，其他人可斟酌時間成本。"),
        (0.0, 3.0, "多數觀眾直接跳過較佳，可把時間留給更合胃口的作品。"),
    ]
    for lower, upper, message in buckets:
        if lower <= score < upper:
            return message
    return "依個人品味自行斟酌。"


def analyze_review(review: str, diversity: float, keyword_top_n: int):
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
    keywords = select_keywords(tokens, keyword_top_n)
    rating_score = rating_from_sentiment(compound)
    return {
        "sentiment_label": sentiment_label,
        "topic": topic,
        "summary": summary,
        "sentiment_score_10": calc_intensity(compound),
        "key_sentences": highlight_sentences,
        "keywords": keywords,
        "rating_pred_10": rating_score,
        "audience_suitability": audience_suggestion_from_rating(rating_score),
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
        initial_sidebar_state="expanded",
    )
    st.title("🎬 IMDB 影評 7 合 1 智能分析")
    st.caption(
        "多類情緒、主題、情緒保留摘要、強度、關鍵句/詞、評分與觀眾適配，一次完成。"
    )

    dataset_rows = load_review_dataset()
    filtered_rows = dataset_rows
    chart_field = "sentiment"
    chart_use_filter = True
    show_raw_table = False
    default_text = (
        "The film's pacing is uneven, but the acting is heartfelt. "
        "I laughed a few times, yet the ending felt rushed and predictable. "
        "Overall, it's a decent weekend watch, nothing mind-blowing."
    )
    if "input_review" not in st.session_state:
        st.session_state.input_review = default_text

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
        keyword_top_n = st.slider(
            "顯示的關鍵字數量",
            min_value=3,
            max_value=10,
            value=6,
            step=1,
        )
        st.markdown(
            f"模型：`{HEURISTICS_FILE}`（本地規則/字典，無需雲端下載）"
        )
        st.markdown(
            f"資料集：`{DATASET_FILE if DATASET_FILE.exists() else '尚未提供'}`"
        )
        st.divider()

        st.subheader("測試素材工具")
        dataset_count = len(dataset_rows)
        st.metric("內建影評數", dataset_count)
        if dataset_rows:
            sentiments = Counter(row["sentiment"] for row in dataset_rows)
            top_sentiments = ", ".join(
                f"{label}:{count}" for label, count in sentiments.most_common(3)
            )
            st.caption(f"常見情緒：{top_sentiments or 'N/A'}")
            sentiment_options = ["全部"] + sorted(sentiments.keys())
            sentiment_filter = st.selectbox(
                "情緒篩選", sentiment_options, key="sentiment_filter"
            )
            rating_min, rating_max = st.slider(
                "評分區間",
                min_value=0.0,
                max_value=10.0,
                value=(0.0, 10.0),
                step=0.5,
            )
            filtered_rows = [
                row
                for row in dataset_rows
                if (sentiment_filter == "全部" or row["sentiment"] == sentiment_filter)
                and rating_min <= float(row.get("rating", 0)) <= rating_max
            ]
            if not filtered_rows:
                st.info("篩選條件下沒有對應的影評，將改用全部資料。")
            sample_source = filtered_rows if filtered_rows else dataset_rows

            sample_map = {
                f"{row.get('id','?')} · {row.get('title','Untitled')}": row
                for row in sample_source
            }
            sample_label = st.selectbox(
                "挑選內建影評",
                list(sample_map.keys()),
                key="sample_selector",
            )
            chosen_sample = sample_map[sample_label]
            st.caption(
                f"標題：{chosen_sample.get('title','Untitled')} · 情緒：{chosen_sample['sentiment']} · 主題：{chosen_sample['topic']} · 評分：{chosen_sample['rating']}"
            )
            col_load, col_random = st.columns(2)
            col_load.button(
                "載入選擇項",
                use_container_width=True,
                key="btn_load_selected",
                on_click=set_review_text,
                args=(chosen_sample["review"],),
            )
            col_random.button(
                "隨機抽樣影評",
                use_container_width=True,
                key="btn_random_sample",
                on_click=set_random_review,
                args=(sample_source,),
            )
            with st.expander("預覽選定影評"):
                st.markdown(f"**{chosen_sample.get('title','Untitled')}**")
                st.write(chosen_sample["review"])
            dataset_text = "\n".join(
                json.dumps(row, ensure_ascii=False) for row in dataset_rows
            )
            st.download_button(
                "下載樣本 JSONL",
                data=dataset_text,
                file_name="imdb_reviews_samples.jsonl",
                mime="application/json",
                use_container_width=True,
            )
            chart_field = st.radio(
                "資料視覺化項目",
                options=["sentiment", "topic"],
                format_func=lambda x: "情緒分佈" if x == "sentiment" else "主題分佈",
                key="chart_field_radio",
            )
            chart_use_filter = st.checkbox(
                "圖表使用篩選結果", value=True, key="chart_use_filter"
            )
            show_raw_table = st.checkbox(
                "顯示資料表", value=False, key="show_raw_table"
            )
        else:
            st.info("尚未找到 `IMDb影評/reviews.jsonl`，僅能手動輸入影評。")

    review = st.text_area(
        "貼上 IMDB 影評文字",
        key="input_review",
        height=200,
        placeholder="輸入英文或中英混合影評，按下分析開始。",
    )

    quick_source = filtered_rows if filtered_rows else dataset_rows
    if quick_source:
        st.caption("快速套用測試影評：")
        quick_samples = []
        sentiments_to_show = ["Positive", "Negative", "Neutral"]
        for sentiment in sentiments_to_show:
            match = next(
                (row for row in quick_source if row["sentiment"] == sentiment),
                None,
            )
            if match:
                quick_samples.append(match)
        if not quick_samples:
            quick_samples = quick_source[: min(3, len(quick_source))]
        cols = st.columns(len(quick_samples))
        for col, sample in zip(cols, quick_samples):
            label = f"{sample.get('title','Untitled')} ({sample['sentiment']})"
            col.button(
                label,
                use_container_width=True,
                key=f"quick_{sample['id']}",
                on_click=set_review_text,
                args=(sample["review"],),
            )

    if st.button("開始分析", type="primary"):
        if not review.strip():
            st.warning("請先輸入影評文字。")
        else:
            with st.spinner("本地模型分析中，請稍候..."):
                parsed = analyze_review(review, temperature, keyword_top_n)
            render_results(parsed)

    chart_source = []
    if dataset_rows:
        chart_source = quick_source if chart_use_filter else dataset_rows
        chart_df = build_chart_dataframe(chart_source, chart_field)
        chart_labels = {"sentiment": "情緒分佈", "topic": "主題分佈"}
        if chart_df is not None:
            title_suffix = chart_labels.get(chart_field, chart_field)
            st.subheader(f"內建影評資料圖 · {title_suffix}")
            chart_data = chart_df.rename(columns={"label": "分類", "count": "篇數"})
            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("分類:N", sort="-y", title="分類"),
                    y=alt.Y("篇數:Q", title="篇數"),
                    color=alt.Color(
                        "分類:N",
                        legend=None,
                        scale=alt.Scale(scheme="teals"),
                    ),
                    tooltip=["分類:N", "篇數:Q"],
                )
                .properties(height=320, title=f"內建影評資料圖 · {title_suffix}")
            )
            st.altair_chart(chart, use_container_width=True)
        if show_raw_table and chart_source:
            st.dataframe(
                pd.DataFrame(chart_source)[
                    ["id", "title", "sentiment", "topic", "rating"]
                ]
            )


if __name__ == "__main__":
    main()
