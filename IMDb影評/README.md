# IMDb影評資料集

此資料夾存放可供離線測試的 IMDb 影評樣本。每筆資料使用 JSON Lines (`reviews.jsonl`) 格式，方便以串流方式讀取。

## 欄位說明

- `id`：自訂的影評識別碼。
- `title`：影評標題或摘要。
- `rating`：評論者給出的 0–10 分評分（浮點數）。
- `sentiment`：人工標記的整體情緒（Positive/Negative/Neutral 等）。
- `topic`：評論焦點（Plot、Acting、Pacing…）。
- `review`：完整文字內容。

## 目前檔案

- `reviews.jsonl`：10 筆範例影評，涵蓋不同情緒與主題，適合餵給 Streamlit App 或其他 NLP 原型。

若需要擴充，請維持 JSON Lines 格式（每行一筆 JSON 物件），並避免洩漏隱私或受版權保護的原文。***
