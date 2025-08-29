import os
import glob
import pandas as pd
import streamlit as st

from src.parser import parse_resume
from src.scoring import overall_score
from src.improve_bot import gap_analysis, suggestions_from_gaps, suggest_improvements
from src.utils import load_role_profile

# ------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.markdown(
    """
<style>
/* Slightly wider modal body for readability */
.stModal > div { max-width: 760px; }
</style>
""",
    unsafe_allow_html=True,
)
st.title("📄 AI-Powered Resume Analyzer")
st.caption("Upload resumes, compare them to job roles, get ranking scores, and receive personalized feedback.")

# ------------------------------------------------------------------------------
# Onboarding helpers
# ------------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("onboarded", False)
    st.session_state.setdefault("show_tour", False)
    st.session_state.setdefault("demo_loaded", False)

def show_onboarding_modal():
    if st.session_state.get("onboarded"):
        return
    try:
        with st.modal("👋 Welcome to AI-Powered Resume Analyzer", key="onboarding"):
            st.write("""
**What you can do here**

1. **Upload** one or more resumes (PDF/TXT).
2. Pick a **role template** (from YAMLs) or **paste a job description**.
3. See **scores, coverage, and suggestions** — then **download CSV**.
4. Use **Multi-Role Comparison** to find the best-fit role for one resume.
            """)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Load demo", type="primary", key="onb_demo"):
                    st.session_state["demo_loaded"] = True
                    st.session_state["onboarded"] = True
                    st.rerun()
            with c2:
                if st.button("Take quick tour", key="onb_tour"):
                    st.session_state["show_tour"] = True
                    st.session_state["onboarded"] = True
                    st.rerun()
            with c3:
                if st.button("Skip", key="onb_skip"):
                    st.session_state["onboarded"] = True
                    st.toast("You can open the tour from the sidebar anytime.")
                    st.rerun()
    except Exception:
        with st.expander("👋 Welcome! Click to see what this app does.", expanded=True):
            st.write(
                "Upload resumes, pick a role or paste a JD, get scores & suggestions. "
                "Use **Multi-Role** to compare one resume across all roles."
            )
            if st.button("Got it"):
                st.session_state["onboarded"] = True

def sidebar_tour_popover():
    with st.sidebar.popover("❓ Need a tour?", use_container_width=True):
        st.markdown("""
- **Upload & Ranking** → upload, select role/JD, get scores  
- **Resume Feedback** → targeted tips for one resume  
- **Multi-Role** → compare one resume across all roles  
- **About** → what it does & tech stack
        """)
        if st.button("Start tour", key="tour_btn"):
            st.session_state["show_tour"] = True
            st.rerun()

def run_quick_tour():
    if not st.session_state.get("show_tour"):
        return
    st.info("➡️ Start at **Upload & Ranking**: upload 2–3 resumes and select *Role template*. "
            "Scroll to the results table, click a row, and read suggestions.")
    if st.button("Got it", key="tour_done"):
        st.session_state["show_tour"] = False
        st.toast("Tour closed. You can reopen it from the sidebar.")
        st.rerun()


def load_demo_inputs():
    """Returns (jd_text, role_name, demo_texts:list[str]) if demo requested, else (None, None, None)."""
    if not st.session_state.get("demo_loaded"):
        return None, None, None
    role_name = "data_scientist"
    jd_text = (
        "We need a Data Scientist with Python, SQL, statistics, and experience "
        "building ML models for production."
    )
    demo_texts = [
        "Experienced analyst proficient in Excel, SQL, and Tableau. Built dashboards and reports; basic Python.",
        "Python developer with scikit-learn, pandas, and model deployment experience; strong SQL and statistics."
    ]
    return jd_text, role_name, demo_texts

# ------------------------------------------------------------------------------
# Role helpers
# ------------------------------------------------------------------------------
def _pretty(name: str) -> str:
    return name.replace("_", " ").title()

def list_role_names(role_dir: str = "data/role_profiles") -> list[str]:
    """Return role keys (filename stems) by scanning YAML files."""
    patterns = [os.path.join(role_dir, "*.yml"), os.path.join(role_dir, "*.yaml")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    names = {os.path.splitext(os.path.basename(p))[0] for p in files}
    return sorted(names)

# ------------------------------------------------------------------------------
# Init onboarding & tour
# ------------------------------------------------------------------------------
init_state()
show_onboarding_modal()
sidebar_tour_popover()
run_quick_tour()

# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Job/Role")
    choice = st.selectbox("Use", ["Role template", "Paste job description"])

    role_name = None
    job_desc = ""

    if choice == "Role template":
        roles = list_role_names()
        if not roles:
            st.warning("No role profiles found in data/role_profiles/*.yml or *.yaml")
            roles = ["data_scientist"]  # fallback
        role_name = st.selectbox("Role", roles, format_func=_pretty)
    else:
        job_desc = st.text_area("Job description", height=180, placeholder="Paste the JD here...")

    st.markdown("---")
    st.header("Weights (advanced)")
    kw_must = st.slider("Keyword (must-have) weight", 0.0, 1.0, 0.45, 0.05)
    kw_nice = st.slider("Keyword (nice-to-have) weight", 0.0, 1.0, 0.15, 0.05)
    emb_w = st.slider("Semantic (embeddings/TF-IDF) weight", 0.0, 1.0, 0.40, 0.05)

# Build JD text / role keywords (default from choice)
profile = None
if choice == "Role template" and role_name:
    profile = load_role_profile(role_name, ".")
    must = profile["keywords"]["must_have"]
    nice = profile["keywords"]["nice_to_have"]
    jd_text = profile["description"] + "\n" + " ".join(must + nice)
else:
    jd_text = job_desc
    must, nice = [], []

# If the user loaded the demo, override JD + role
demo_jd, demo_role, demo_texts = load_demo_inputs()
if demo_jd:
    jd_text = demo_jd
    role_name = demo_role
    # If we have the role profile locally, use its keywords for gap analysis
    try:
        demo_profile = load_role_profile(role_name, ".")
        must = demo_profile["keywords"]["must_have"]
        nice = demo_profile["keywords"]["nice_to_have"]
    except Exception:
        pass

weights = {"keyword_must": kw_must, "keyword_nice": kw_nice, "embeddings": emb_w}

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Upload & Ranking", "💡 Resume Feedback", "ℹ️ About", "🔀 Multi-Role Comparison"
])

# =============================
# TAB 1: Upload & Ranking
# =============================
with tab1:
    st.header("Upload Resumes & Rank")

    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True
    )

    # Empty-state hint
    if not uploaded_files and not demo_texts:
        st.info("💡 Tip: drag & drop PDFs here, or click **Load demo** in the onboarding to try it instantly.")

    rows = []

    # Demo path: build rows from demo text snippets
    if demo_texts and jd_text:
        for i, txt in enumerate(demo_texts, start=1):
            res_text = txt
            sc = overall_score(res_text, jd_text, weights, must, nice)
            gaps = gap_analysis(res_text, must, nice) if (must or nice) else {"missing_must": [], "missing_nice": []}
            rows.append({
                "filename": f"demo_resume_{i}.txt",
                **sc,
                "missing_must": ", ".join(gaps.get("missing_must", [])),
                "missing_nice": ", ".join(gaps.get("missing_nice", [])),
            })

    # Regular upload path
    elif uploaded_files and jd_text:
        for f in uploaded_files:
            kind = "pdf" if f.type == "application/pdf" or f.name.lower().endswith(".pdf") else "txt"
            data = f.read()
            parsed = parse_resume(data, kind)
            res_text = parsed["raw_text"]
            sc = overall_score(res_text, jd_text, weights, must, nice)
            gaps = gap_analysis(res_text, must, nice) if (must or nice) else {"missing_must": [], "missing_nice": []}
            rows.append({
                "filename": f.name,
                **sc,
                "missing_must": ", ".join(gaps.get("missing_must", [])),
                "missing_nice": ", ".join(gaps.get("missing_nice", [])),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
        st.toast("Scoring complete. Scroll for results.")
        st.subheader("📊 Ranked Candidates")
        st.dataframe(df, use_container_width=True)

        # Progress + metric for top candidate
        top = df.iloc[0]
        st.subheader(f"Best Match: {top['filename']}")
        st.metric("Total Score", f"{top['total_score']:.2f}")
        st.progress(min(1.0, float(top["total_score"])) if pd.notna(top["total_score"]) else 0.0)
        if float(top["total_score"]) >= 0.85:
            st.balloons()

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results (CSV)", csv, "resume_ranking.csv", "text/csv")

        # Details & Suggestions
        st.subheader("Details & Suggestions")
        pick = st.selectbox("Select a resume to review", df["filename"].tolist())
        current = df[df["filename"] == pick].iloc[0]

        st.markdown(
            f"**Score:** {current['total_score']:.3f} · "
            f"**Semantic:** {current.get('semantic_sim', 0):.3f} · "
            f"**Must cov.:** {current.get('must_coverage', 0):.2f} · "
            f"**Nice cov.:** {current.get('nice_coverage', 0):.2f}"
        )

        if profile or demo_jd:
            gaps = {
                "missing_must": current["missing_must"].split(", ") if current["missing_must"] else [],
                "missing_nice": current["missing_nice"].split(", ") if current["missing_nice"] else [],
            }
            tips = suggestions_from_gaps(gaps)
            if tips:
                st.markdown("**How to improve this resume for the selected role:**")
                for t in tips:
                    st.write(f"- {t}")
    elif jd_text and not (uploaded_files or demo_texts):
        st.info("⬆️ Upload one or more resumes to compute rankings.")
    else:
        st.info("⬆️ Upload resumes and paste a job description or pick a role template.")

# =============================
# TAB 2: Resume Feedback
# =============================
with tab2:
    st.header("💡 Resume Improvement Suggestions")
    file = st.file_uploader("Upload a resume (PDF/TXT)", type=["pdf", "txt"], key="feedback")
    target_role = st.text_input("Target Role (e.g., Data Scientist, ML Engineer, Data Analyst, Software Developer)")

    if file and target_role:
        kind = "pdf" if file.type == "application/pdf" or file.name.lower().endswith(".pdf") else "txt"
        data = file.read()
        parsed = parse_resume(data, kind)
        res_text = parsed["raw_text"]
        suggestions = suggest_improvements(res_text, target_role)

        st.subheader("🔎 Feedback")
        for tip in suggestions:
            st.write(f"- {tip}")
    else:
        st.info("⬆️ Upload a single resume and enter the target role to get feedback.")

# =============================
# TAB 3: About
# =============================
with tab3:
    st.header("ℹ️ About this Project")
    st.markdown("""
### AI Resume Analyzer
This app parses resumes, compares them with job descriptions, and ranks candidates based on:
- ✅ **Keyword coverage** (must-have and nice-to-have skills)
- 🤖 **Semantic similarity** using HuggingFace embeddings
- 📊 **Weighted scoring** with adjustable components

**Features**
- Multi-resume upload and ranking  
- Progress bars and metrics for clarity  
- Downloadable CSV of candidate scores  
- Tailored resume improvement suggestions  
- Auto-discovered role profiles (YAML)  

**Tech Stack**
- Python, Streamlit  
- NLTK, spaCy  
- HuggingFace Sentence-Transformers  
- Pandas, scikit-learn  

---
✨ Built by Harsh Sharma — showcasing NLP + AI for smarter resume analysis.
""")

# =============================
# TAB 4: Multi-Role Comparison (auto-detect roles)
# =============================
with tab4:
    st.header("🔀 Multi-Role Comparison")
    file = st.file_uploader("Upload a single resume (PDF/TXT)", type=["pdf", "txt"], key="multirole")

    if not file:
        st.info("⬆️ Upload one resume to see how it fits **all** available roles.")
    else:
        roles = list_role_names()
        if not roles:
            st.warning("No role profiles found in data/role_profiles/*.yml or *.yaml")
        else:
            kind = "pdf" if file.type == "application/pdf" or file.name.lower().endswith(".pdf") else "txt"
            data = file.read()
            parsed = parse_resume(data, kind)
            res_text = parsed["raw_text"]

            results = []
            for r in roles:
                prof = load_role_profile(r, ".")
                must_r = prof["keywords"]["must_have"]
                nice_r = prof["keywords"]["nice_to_have"]
                jd_r = prof["description"] + "\n" + " ".join(must_r + nice_r)
                sc = overall_score(res_text, jd_r, weights, must_r, nice_r)
                results.append({"Role": _pretty(r), "Score": sc["total_score"], **sc})

            df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
            st.subheader("📊 Role Fit Comparison")
            st.dataframe(df, use_container_width=True)

            best = df.iloc[0]
            st.success(f"✅ Best fit: **{best['Role']}** with score {best['Score']:.2f}")
