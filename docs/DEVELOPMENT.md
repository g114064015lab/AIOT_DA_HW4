## Development Guide

### 1. Prerequisites

- Python 3.11 (match `runtime.txt`)
- pip + virtualenv (recommended)
- Git
- Optional: `make` for scripted tasks

### 2. Environment Setup

```bash
python3.11 -m venv .venv
.venv/Scripts/activate  # Windows
# or source .venv/bin/activate on macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements intentionally avoid GPU-only dependencies so the app can run in CPU-only environments such as Streamlit Cloud.

### 3. Running the App

```bash
streamlit run app.py
```

Key runtime configs:
- The sidebar sliders control the heuristic diversity and keyword count.
- Quick-pick buttons always load the best positive, neutral, and negative samples based on dataset ratings.
- Dataset filters operate on `IMDb影評/reviews.jsonl`; edit or expand this file to add more samples.

### 4. Quality Checklist

Before pushing changes:

1. **Static check** – `py -m py_compile app.py` (already used in CI-like steps).
2. **Manual test** – run `streamlit run app.py`, try the quick-pick buttons, filters, and ensure the chart renders.
3. **Docs** – update `readme.md` or files in `docs/` when modifying architecture or workflows.
4. **Git hygiene** – commit only relevant files, keep credentials out of the repo, avoid large binary blobs.

### 5. Adding Data or Heuristics

- **New Reviews:** append JSON lines to `IMDb影評/reviews.jsonl` with `id`, `title`, `rating`, `sentiment`, `topic`, and `review`.
- **Keyword Tuning:** edit `model/heuristics.json` to add new sentiment cues, topics, or audience personas.
- **Rating Calibration:** no manual step required; the regression refits whenever the dataset cache is refreshed.

### 6. Deployment Notes

- Streamlit Cloud follows the files in `main`. After pushing:
  1. Visit the app URL.
  2. Click “Rerun” if the deployment did not restart automatically.
  3. Check logs via “Manage app” if dependency/install issues appear.
- For local Docker deployments, wrap `streamlit run app.py` inside a slim Python base image and mount the repo volume.

### 7. Branching & Contributions

- Use feature branches named `feat/...`, `fix/...`, or `docs/...`.
- Open PRs summarizing the change, tests performed, and any doc updates.
- Keep `main` deployable; avoid pushing experimental code without validation.

Refer back to `docs/ARCHITECTURE.md` for a system overview, and `REPORT.md` for the abstract/summary used in course submissions.
