# AI-Powered Resume Analyzer

**What it does**
- Parse resumes (PDF/TXT), compute keyword coverage and semantic similarity to a job/role, rank candidates, and suggest improvements.
- Includes a lightweight chatbot pane with practical tips for Data Science resumes.

**Tech**
- Python, Streamlit
- sentence-transformers (embeddings), TF‑IDF (semantic overlap)
- NLTK, spaCy (optional), PyPDF2

**Run locally**
```bash
python -m venv .venv
# activate venv (Windows)
.venv\Scripts\activate
# or (macOS/Linux)
source .venv/bin/activate

pip install -r requirements.txt
python setup_nlp.py
# optional spaCy model
python -m spacy download en_core_web_sm

streamlit run app.py
```

**Notes**
- First run will download `all-MiniLM-L6-v2` from HuggingFace.
- Adjust scoring weights in the sidebar.
- Add more roles by dropping YAML files into `data/role_profiles/`.
