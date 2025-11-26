# 📘 IMDB 情意分析延伸功能 — README

本專案以 IMDB 電影評論資料集為基礎，除了基本的「正負評二元分類」之外，更進一步延伸出 **七項強化情意分析的功能**，讓模型不只是分類器，而是能夠深入理解評論、萃取資訊並產生應用價值。

以下文件將逐項介紹功能目的、方法與對應可用 Prompt。

---

# 🚀 快速安裝與執行

1. **建議使用 Python 3.11**（PyTorch 目前尚未穩定支援 3.13）。
2. 建立虛擬環境並安裝依賴：
   ```bash
   python3.11 -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install --upgrade pip
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   若安裝 torch 失敗，可再嘗試：
   ```bash
   pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
   pip install transformers==4.44.2 sentencepiece
   ```
3. 執行：
   ```bash
   streamlit run app.py
   ```

---

# 📂 功能總覽

| 編號 | 功能名稱 | 說明 |
|------|----------|------|
| 1 | 多分類情緒分類 | 從二元分類擴展成多種情緒標籤 |
| 2 | 影評主題分類 | 抽取評論討論的主題（演技/劇情/特效…） |
| 3 | 影評摘要生成 | 產生保持情緒的短摘要 |
| 4 | 情緒強度分析 | 評估情緒強度（1–10）與原因 |
| 5 | 關鍵句與關鍵字抽取 | 找出最能代表情緒的句子與詞 |
| 6 | 評分推估 | 依評論推測可能給的星等（1–10） |
| 7 | 觀眾類型建議 | 依評論推論適合與不適合的觀眾類型 |

---

# 🧩 1. 多分類情緒分類（Multi-class Sentiment Classification）

### 📘 功能說明
傳統 IMDB 任務僅提供 **正評 vs 負評** 的標籤。  
本功能擴展成更多情緒類別，如：

- 正面（Positive）
- 中立（Neutral）
- 負面（Negative）
- 感動（Touched）
- 生氣（Angry）
- 失望（Disappointed）
- 驚喜（Surprised）

能讓分析結果更細緻，適合應用於行銷、品牌聲量、公眾輿情探勘等場景。



---

# 🧩 2. 影評主題分類（Review Topic Classification）

### 📘 功能說明
影評通常包含多面向的評論，本功能用來判斷評論者到底在談：

- 劇情（Plot）
- 演技（Acting）
- 導演（Directing）
- 特效（Visual Effects）
- 配樂（Music）
- 節奏（Pacing）
- 其他（Other）

可用於更精細的評論分析、電影推薦系統或內容摘要生成。



---

# 🧩 3. 影評摘要生成（Sentiment-preserving Summarization）

### 📘 功能說明
將冗長影評壓縮成 1–2 句摘要，但 **保留原本情緒**。  
讓使用者快速理解評論內容與情緒色彩，適合平台側摘要評論、推薦系統顯示精華文字等用途。


---

# 🧩 4. 情緒強度分析（Sentiment Intensity Scoring）

### 📘 功能說明
除了知道評論正負，還能判斷情感的「強烈程度」。  
例如：

- 正面情緒：8/10（非常開心）
- 負面情緒：3/10（略為不滿）
- 負面情緒：10/10（極度憤怒）

適用於：

- 客服情緒監測  
- 影評極端情緒偵測  
- 討論串情緒熱度分析  



---

# 🧩 5. 關鍵句與關鍵字抽取（Key Sentence & Keyword Extraction）

### 📘 功能說明
從影評中找出：

- 最能表達情感的句子  
- 描述電影的核心字彙  

有助於：

- 搜尋引擎 SEO 分析  
- 摘要呈現  
- 觀眾情緒推斷  


---

# 🧩 6. 平均評分推估（Rating Prediction from Reviews）

### 📘 功能說明
從影評推斷評論者可能會給電影的評分（1–10）。  
例如：

- 「劇情無聊死了」 → 2/10  
- 「完美的演出與配樂」 → 9/10  

可用於：

- 電影推薦模型  
- 異常評論偵測  
- 影評自動標準化  


---

# 🧩 7. 觀眾類型建議（Audience Suitability Analysis）

### 📘 功能說明
從影評推論：

- 這部電影適合哪些觀眾？  
- 哪些人可能不會喜歡？  

例如：

- 喜歡沉重劇情的人適合  
- 喜歡輕鬆喜劇的人可能不適合  

這種分析適合推薦系統、行銷策略調整、受眾分群等應用。

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Project Abstract](REPORT.md)

