# app.py
# -*- coding: utf-8 -*-
#
# Streamlit 應用：針對 IMDB 影評，實作 README 中的 1~7 功能
#
# 需先安裝：
#   pip install streamlit transformers sentencepiece

import re
import string
from collections import Counter

import streamlit as st
from transformers import pipeline


# -----------------------------
# 初始化模型（避免重複載入）
# -----------------------------
@st.cache_resource
def load_zero_shot_classifier():
    # 用於多分類情緒 & 主題分類
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@st.cache_resource
def load_sentiment_model():
    # 二元情緒模型，用於情緒強度與評分推估
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource
def load_summarizer():
    # 摘要模型
    return pipeline("summarization", model="facebook/bart-large-cnn")


zero_shot_clf = load_zero_shot_classifier()
sentiment_clf = load_sentiment_model()
summarizer = load_summarizer()


# -----------------------------
# 工具函式
# -----------------------------
def split_sentences(text: str):
    # 簡單句子切分
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    # 過濾太短的片段
    return [s.strip() for s in parts if len(s.strip()) > 10]


def extract_keywords(text: str, top_k: int = 5):
    # 簡單的關鍵字抽取：去除停用詞 + 統計頻率
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "for",
        "to",
        "of",
        "is",
        "are",
        "was",
        "were",
        "it",
        "this",
        "that",
        "with",
        "as",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "my",
        "your",
        "their",
        "our",
        "me",
        "him",
        "her",
        "them",
        "very",
        "really",
        "just",
        "so",
        "too",
        "also",
    }
    text = text.lower()
    # 去除標點
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    counter = Counter(tokens)
    return [w for w, _ in counter.most_common(top_k)]


def scale_sentiment_to_intensity(label: str, score: float) -> int:
    # 將情緒分數映射到 1~10 強度
    # positive → score 越高強度越高
    # negative → score 越高強度越高
    # 這裡將 0~1 線性轉成 1~10
    intensity = int(round(score * 9 + 1))
    intensity = max(1, min(10, intensity))
    return intensity


def sentiment_to_rating(label: str, score: float) -> int:
    # 根據正負情緒估計 1~10 評分
    # 正向：基準 6~10，負向：1~5
    if label.upper() == "POSITIVE":
        rating = 6 + score * 4  # 6~10
    else:
        rating = 1 + (1 - score) * 4  # 1~5
    rating = int(round(rating))
    rating = max(1, min(10, rating))
    return rating


# -----------------------------
# 功能 1：多分類情緒分類
# -----------------------------
def func_multiclass_sentiment(review_text: str):
    labels = ["positive", "neutral", "negative", "touched", "angry", "disappointed", "surprised"]
    result = zero_shot_clf(review_text, candidate_labels=labels, multi_label=False)
    st.subheader("1️⃣ 多分類情緒分類結果")

    # 排序顯示
    scores = list(zip(result["labels"], result["scores"]))
    scores.sort(key=lambda x: x[1], reverse=True)

    st.write("**預測情緒標籤（由高到低）：**")
    for label, score in scores:
        st.write(f"- {label}（score = {score:.3f}）")


# -----------------------------
# 功能 2：影評主題分類
# -----------------------------
def func_topic_classification(review_text: str):
    labels = ["Plot", "Acting", "Directing", "Visual Effects", "Music", "Pacing", "Other"]
    result = zero_shot_clf(review_text, candidate_labels=labels, multi_label=False)
    st.subheader("2️⃣ 影評主題分類結果")

    scores = list(zip(result["labels"], result["scores"]))
    scores.sort(key=lambda x: x[1], reverse=True)

    st.write("**預測主題（由高到低）：**")
    for label, score in scores:
        st.write(f"- {label}（score = {score:.3f}）")


# -----------------------------
# 功能 3：影評摘要生成
# -----------------------------
def func_summarization(review_text: str):
    st.subheader("3️⃣ 影評摘要生成結果")
    # 適當控制長度
    max_len = 130
    min_len = 30
    # 太短就沒必要摘要
    if len(review_text.split()) < 40:
        st.info("影評略短，直接顯示原文：")
        st.write(review_text)
        return

    summary = summarizer(
        review_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
    )[0]["summary_text"]

    st.write("**摘要結果：**")
    st.write(summary)


# -----------------------------
# 功能 4：情緒強度分析
# -----------------------------
def func_sentiment_intensity(review_text: str):
    st.subheader("4️⃣ 情緒強度分析結果")

    result = sentiment_clf(review_text)[0]
    label = result["label"]  # POSITIVE / NEGATIVE
    score = float(result["score"])
    intensity = scale_sentiment_to_intensity(label, score)

    sentiment_zh = "正面" if label.upper() == "POSITIVE" else "負面"

    st.write(f"**感受：** {sentiment_zh} ({label})")
    st.write(f"**模型信心分數：** {score:.3f}")
    st.write(f"**推定情緒強度（1–10）：** {intensity}")
    st.write("**說明：** 強度是根據情緒分類模型的信心分數，線性映射到 1–10 的區間。")


# -----------------------------
# 功能 5：關鍵句與關鍵字抽取
# -----------------------------
def func_key_sentences_keywords(review_text: str):
    st.subheader("5️⃣ 關鍵句與關鍵字抽取結果")

    sentences = split_sentences(review_text)
    if not sentences:
        st.warning("無法從文字中切分出有效句子。")
        return

    # 簡單依句子長度排序，取前 3 句
    sentences_sorted = sorted(sentences, key=len, reverse=True)
    top_sentences = sentences_sorted[:3]

    st.write("**關鍵句（最多三句）：**")
    for i, s in enumerate(top_sentences, 1):
        st.write(f"{i}. {s}")

    keywords = extract_keywords(review_text, top_k=5)
    st.write("**關鍵字（最多五個）：**")
    st.write(", ".join(keywords) if keywords else "（無明顯關鍵字）")


# -----------------------------
# 功能 6：評分推估
# -----------------------------
def func_rating_prediction(review_text: str):
    st.subheader("6️⃣ 評分推估結果")

    result = sentiment_clf(review_text)[0]
    label = result["label"]
    score = float(result["score"])

    rating = sentiment_to_rating(label, score)
    sentiment_zh = "正面" if label.upper() == "POSITIVE" else "負面"

    st.write(f"**情緒判定：** {sentiment_zh} ({label}), score = {score:.3f}")
    st.write(f"**推估評分（1–10）：** {rating}")
    st.write("**說明：** 正面情緒對應 6–10 分區間，負面情緒對應 1–5 分區間，再依模型信心分數調整。")


# -----------------------------
# 功能 7：觀眾類型建議
# -----------------------------
def func_audience_suggestion(review_text: str):
    st.subheader("7️⃣ 觀眾類型建議結果")

    result = sentiment_clf(review_text)[0]
    label = result["label"]
    score = float(result["score"])
    sentiment_zh = "正面" if label.upper() == "POSITIVE" else "負面"

    rating = sentiment_to_rating(label, score)

    st.write(f"**情緒判定：** {sentiment_zh} ({label}), score = {score:.3f}")
    st.write(f"**推估評分：** {rating}/10")

    st.write("**適合的觀眾類型（推論）：**")
    if label.upper() == "POSITIVE":
        st.write("- 喜歡這種類型題材的觀眾。")
        st.write("- 對演員或導演已有好感的影迷。")
        st.write("- 接受片中節奏與敘事風格的觀眾。")
    else:
        st.write("- 不喜歡節奏拖沓或劇情薄弱的觀眾應謹慎觀看。")
        st.write("- 對演員或導演原本期待很高的人可能會失望。")
        st.write("- 比較在意劇情合理性、剪輯流暢度的觀眾可能不適合。")

    st.write("**不適合 / 可能不喜歡的觀眾（推論）：**")
    if label.upper() == "POSITIVE":
        st.write("- 對此題材完全不感興趣的人，可能仍不會特別喜歡。")
        st.write("- 偏好節奏極快、爆米花類電影的觀眾，若本片較內斂，可能覺得無聊。")
    else:
        st.write("- 極度在意片長與節奏的觀眾。")
        st.write("- 期待強烈動作場面或高張力劇情，但本片較平淡的觀眾。")


# -----------------------------
# Streamlit 介面
# -----------------------------
st.set_page_config(
    page_title="IMDB 情意分析工具",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 IMDB 影評情意分析 — 功能 1~7 Demo")

st.markdown(
    "請輸入一段 IMDB 影評文字，並選擇要執行的功能。"
)

st.write("---")

col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area(
        "輸入影評（英文為主）：",
        height=250,
        placeholder=(
            "例如：This movie was absolutely fantastic. The performances were top-notch and the story "
            "kept me engaged from start to finish..."
        ),
    )

with col2:
    func_choice = st.radio(
        "選擇功能：",
        (
            "1. 多分類情緒分類",
            "2. 影評主題分類",
            "3. 影評摘要生成",
            "4. 情緒強度分析",
            "5. 關鍵句與關鍵字抽取",
            "6. 評分推估",
            "7. 觀眾類型建議",
        ),
    )
    run_button = st.button("🚀 執行分析")

st.write("---")

if run_button:
    if not review_text.strip():
        st.warning("請先輸入一段影評再執行分析。")
    else:
        if func_choice.startswith("1"):
            func_multiclass_sentiment(review_text.strip())
        elif func_choice.startswith("2"):
            func_topic_classification(review_text.strip())
        elif func_choice.startswith("3"):
            func_summarization(review_text.strip())
        elif func_choice.startswith("4"):
            func_sentiment_intensity(review_text.strip())
        elif func_choice.startswith("5"):
            func_key_sentences_keywords(review_text.strip())
        elif func_choice.startswith("6"):
            func_rating_prediction(review_text.strip())
        elif func_choice.startswith("7"):
            func_audience_suggestion(review_text.strip())
