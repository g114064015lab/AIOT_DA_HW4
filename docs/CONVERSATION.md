## Conversation Log

Each entry notes the speaker so it’s clear who said what.

### 1. User
> 根據readme，幫我撰寫python code我要架設在streamlit，並上傳到https://github.com/g114064015lab/AIOT_DA_HW4

### 2. Assistant
Explained the new Streamlit app (local Qwen 1.5B instruct), parsing prompts, rendering metrics, and suggested install/test/push steps.

### 3. User
> 你直接幫我push上來:https://github.com/g114064015lab/AIOT_DA_HW4

### 4. Assistant
Committed `app.py`, pushed to `main`, confirmed GitHub status.

### 5. User
Posted pip install failure log showing `tokenizers` build error on Python 3.13.

### 6. Assistant
Advised using Python 3.11 or setting `PYO3_USE_ABI3_FORWARD_COMPATIBILITY`, recommended reinstall steps.

### 7. User
> 直接幫我修好這個問題

### 8. Assistant
Updated `tokenizers` requirement to 0.22.1, committed & pushed fix.

### 9. User
Shared Streamlit Cloud install log showing conflict (`transformers` needed tokenizers <0.20).

### 10. Assistant
Upgraded `transformers` to 4.57.3 to align, pushed change, noted redeploy steps.

### 11. User
Provided runtime `ValueError` (model loading requiring accelerate).

### 12. Assistant
Switched app to lightweight local heuristics (no transformer), added dataset-based analyzer, committed & pushed.

### 13. User
> 無法載入語言模型...你用的模型也要上傳到github...

### 14. Assistant
Explained heuristics approach already ships in repo, introduced dataset folder, new JSON heuristics, etc.

### 15. User
> 再幫我在github建立一個IMDb影評資料夾，裡面都放影評的資料

### 16. Assistant
Created `IMDb影評/` with README and `reviews.jsonl`, pushed to GitHub.

### 17. User
> reviews.jsonl裡面還可以放更多的data

### 18. Assistant
Appended 10 more sample reviews (now 20), committed and pushed.

### 19. User
Reported installation log/new Streamlit error again.

### 20. Assistant
Diagnosed sample button crash, updated sidebar to support filters, dataset viewer, quick buttons, and chart; committed/pushed.

### 21. User
Shared new crash trace after button click.

### 22. Assistant
Added null-safe movie labels, `set_review_text` helper; committed/pushed.

### 23. User
> 電影名稱幫我拿掉好了，然後button為什麼會有n/a字樣?...觀眾適配...

### 24. Assistant
Removed movie name references, ensured button labels show sentiment/topic, introduced rating-based audience guidance, pushed.

### 25. User
Requested documentation and dataset growth (30 more reviews, chart title fix, add report).

### 26. Assistant
Added more reviews (total 50), rating chart styling, quick buttons, created `REPORT.md`, etc., pushing multiple commits.

### 27. User
Noted negative preset still weak; asked for stronger example.

### 28. Assistant
Ensured quick buttons pick highest/lowest rated samples, refined ordering, pushed updates.

### 29. User
Mentioned chart Mermaid parse error.

### 30. Assistant
Fixed Mermaid nodes in `ARCHITECTURE.md`, applied rebase/push.

### 31. User
> https://aiotdahw4-ggligitjqn3czaxftowmyf.streamlit.app/ 幫我把streamlit的連結新增到readme

### 32. Assistant
Added deployment link to README, resolved rebase, pushed.

### 33. User
Requested conversation export as PDF (later asked for GitHub version instead), leading to this log file creation.
