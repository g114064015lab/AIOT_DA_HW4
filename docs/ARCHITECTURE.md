## Architecture Overview

This Streamlit app delivers seven types of IMDB review analysis without relying on large hosted LLMs. The solution is organized into the following layers:

1. **Presentation (Streamlit UI)**
   - `app.py` renders the text area, sidebar controls, chart, and metrics.
   - Widgets feed user input (review text, temperature, keyword count, dataset filters) into the analysis pipeline.
   - Session state keeps the textarea content stable so quick-pick buttons and dataset selectors behave predictably.

2. **Analysis Engine (Heuristic Model)**
   - Built on top of `vaderSentiment` for polarity scoring.
   - Additional classifiers use keyword dictionaries in `model/heuristics.json` to map reviews to topics, fine-grained sentiment labels, and audience personas.
   - Sentence scoring + keyword extraction rely on token frequency, stop words, and sentiment-weighted lengths.
   - Rating prediction = linear mapping between VADER compound scores and the curated dataset’s ground-truth ratings (fitted at runtime via cached regression).

3. **Datasets & Assets**
   - `IMDb影評/reviews.jsonl` provides 50 sample reviews with titles, ratings, sentiments, and topics. The sidebar tooling can filter, preview, download, and load these samples.
   - `model/heuristics.json` centralizes the keyword lists used across sentiments, topics, and audience personas.

4. **Caching & Performance**
   - `@st.cache_resource` keeps the VADER analyzer + heuristics in memory.
   - `@st.cache_data` is used for dataset loading and the rating regression to avoid recomputation between reruns.
   - No files are downloaded at runtime beyond local assets, ensuring the app runs deterministically on Streamlit Cloud.

5. **Extensibility Points**
   - To adjust classification behavior, edit `model/heuristics.json` and rerun the app (Streamlit’s cache can be cleared via the sidebar menu).
   - To recalibrate ratings, add more labeled reviews in `IMDb影評/reviews.jsonl`; the regression automatically picks up the new mapping.
   - For future hybrid models, wrap additional predictors in helper functions and call them from `analyze_review`.

## Data Flow

```mermaid
flowchart TD
    A[User Input / Dataset Button] --> B[Streamlit Session State]
    B --> C[analyze_review()]
    C --> D[VADER Analyzer]
    C --> E[Keyword Heuristics]
    C --> F[Rating Regression Fit]
    D & E & F --> G[Structured JSON Result]
    G --> H[Metrics + Charts + Tables]
```

## Key Design Decisions

- **Heuristics instead of LLMs:** keeps the app lightweight, deterministic, and deployable in environments without GPU access.
- **Embedded dataset & config:** having JSONL and heuristics inside the repo supports offline usage and aligns with reproducible assignments.
- **Transparent UX:** quick-pick buttons show sentiment + rating, charts are labeled “內建影評資料圖,” and error handling uses `st.error`/`st.stop` for clarity.

Refer to `docs/DEVELOPMENT.md` for tips on extending or testing these components.
