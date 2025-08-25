# AI-Powered Resume Analyzer

Upload resumes (PDF/TXT), compare them to a role/JD, get a ranked score, and generate concrete improvement tips.

**Live demo:**  
**Repo:** https://github.com/harshsh-py/ai-resume-analyzer

---

## What it does
- Parses resumes → clean text and sections
- Scores fit using:
  - **Semantic similarity** (HuggingFace `all-MiniLM-L6-v2` + TF-IDF)
  - **Keyword coverage** (must-have / nice-to-have)
- Ranks candidates and explains the score
- “How to improve” suggestions
- Multi-role comparison (e.g., Data Scientist, ML Engineer, Data Analyst)

## UI
- Tabs: **Upload & Ranking**, **Resume Feedback**, **About**, **Multi-Role Comparison**
- Progress bars/metrics for scores
- CSV download of ranked results

## Tech
- Streamlit, Python 3.11
- sentence-transformers, scikit-learn
- NLTK, spaCy (optional)
- PyPDF2 / pdfminer.six
- Pandas, NumPy

## Run locally
```bash
# in repo root
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# or Git Bash
source .venv/Scripts/activate

pip install -r requirements.txt
python setup_nlp.py
python -m spacy download en_core_web_sm  # optional
python -m streamlit run app.py
