## ABSTRACT

### ABSTRACT

**Overview**  
This project delivers a fully offline Streamlit experience that transforms a traditional binary IMDB sentiment task into a seven-in-one analytical assistant for AIOT coursework and rapid prototyping. A custom heuristic pipeline built on VADER sentiment analysis, curated topic and emotion dictionaries, and lightweight text utilities outputs multi-class sentiment labels, topical focus, sentiment-preserving summaries, intensity scores, key sentence extraction, keyword clouds, rating projections, and audience suitability guidance—without relying on hosted large language models.

**Methodology**  
Deterministic behavior on Streamlit Cloud is achieved by replacing transformer downloads with local logic, tuning hyperparameters for CPU-only execution, and turning predicted ratings into rating-band messages so every score maps to a concrete viewing recommendation. The bundled heuristics in `model/heuristics.json` remain editable, encouraging instructors to extend keyword sets or audience segments as class exercises.

**Dataset & UI Enhancements**  
Fifty curated English reviews spanning comedies, dramas, horror shorts, and hybrid genres power the latest sidebar tooling. Users can filter by sentiment or rating range, preview any sample, trigger random loads, download the JSONL file, or press quick-pick buttons—now guaranteed to surface the most positive, most negative, and neutral exemplars by sorting on rating. An Altair visualization titled **「內建影評資料圖」** keeps the dataset transparent with interactive tooltips, while a stateful textarea, adjustable keyword counts, and sensitivity sliders streamline experimentation.

**Outcomes**  
The repository documents setup requirements (Python 3.11, CPU-friendly wheels) and now ships with this formatted abstract for reporting needs. Together, the heuristics, well-scoped dataset, and thoughtful UX affordances approximate richer IMDB analyses without heavyweight models, offering a reproducible template for AIOT labs, hackathons, or constrained deployments that must run reliably within tight resource envelopes.
