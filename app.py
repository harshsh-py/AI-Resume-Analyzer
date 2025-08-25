import os
import glob
import pandas as pd
import streamlit as st

from src.parser import parse_resume
from src.scoring import overall_score
from src.improve_bot import gap_analysis, suggestions_from_gaps, suggest_improvements
from src.utils import load_role_profile


# Page setup

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("Upload resumes, compare them to job roles, get ranking scores, and receive personalized feedback.")


# Helpers

def list_role_names(role_dir: str = "data/role_profiles") -> list[str]:
    """Return role names by scanning YAML files (e.g., data_scientist.yml -> data_scientist)."""
    files = glob.glob(os.path.join(role_dir, "*.yml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in files)


# Sidebar controls

with st.sidebar:
    st.header("Job/Role")
    choice = st.selectbox("Use", ["Role template", "Paste job description"])

    role_name = None
    job_desc = ""

    if choice == "Role template":
        roles = list_role_names()
        if not roles:
            st.warning("No role profiles found in data/role_profiles/*.yml")
            roles = ["data_scientist"]  # fallback
        role_name = st.selectbox("Role", roles)
    else:
        job_desc = st.text_area("Job description", height=180, placeholder="Paste the JD here...")

    st.markdown("---")
    st.header("Weights (advanced)")
    kw_must = st.slider("Keyword (must-have) weight", 0.0, 1.0, 0.45, 0.05)
    kw_nice = st.slider("Keyword (nice-to-have) weight", 0.0, 1.0, 0.15, 0.05)
    emb_w = st.slider("Semantic (embeddings/TF-IDF) weight", 0.0, 1.0, 0.40, 0.05)


profile = None
if choice == "Role template" and role_name:
    profile = load_role_profile(role_name, ".")
    must = profile["keywords"]["must_have"]
    nice = profile["keywords"]["nice_to_have"]
    jd_text = profile["description"] + "\n" + " ".join(must + nice)
else:
    jd_text = job_desc
    must, nice = [], []

weights = {"keyword_must": kw_must, "keyword_nice": kw_nice, "embeddings": emb_w}


# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Upload & Ranking", "💡 Resume Feedback", "ℹ️ About", "🔀 Multi-Role Comparison"
])


# TAB 1: Upload & Ranking

with tab1:
    st.header("Upload Resumes & Rank")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files and jd_text:
        rows = []
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

        df = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)

        st.subheader("📊 Ranked Candidates")
        st.dataframe(df, use_container_width=True)

        # Progress + metric for top candidate
        top = df.iloc[0]
        st.subheader(f"Best Match: {top['filename']}")
        st.metric("Total Score", f"{top['total_score']:.2f}")
        st.progress(min(1.0, float(top["total_score"])) if pd.notna(top["total_score"]) else 0.0)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results (CSV)", csv, "resume_ranking.csv", "text/csv")

        # Details & Suggestions
        st.subheader("Details & Suggestions")
        pick = st.selectbox("Select a resume to review", df["filename"].tolist())
        current = df[df["filename"] == pick].iloc[0]

        st.markdown(
            f"**Score:** {current['total_score']:.3f} · "
            f"**Semantic:** {current['semantic_sim']:.3f} · "
            f"**Must cov.:** {current['must_coverage']:.2f} · "
            f"**Nice cov.:** {current['nice_coverage']:.2f}"
        )

        if profile:
            gaps = {
                "missing_must": current["missing_must"].split(", ") if current["missing_must"] else [],
                "missing_nice": current["missing_nice"].split(", ") if current["missing_nice"] else [],
            }
            tips = suggestions_from_gaps(gaps)
            st.markdown("**How to improve this resume for the selected role:**")
            for t in tips:
                st.write(t)
    else:
        st.info("⬆️ Upload resumes and paste a job description or pick a role template.")


# TAB 2: Resume Feedback

with tab2:
    st.header("💡 Resume Improvement Suggestions")
    file = st.file_uploader("Upload a resume (PDF/TXT)", type=["pdf", "txt"], key="feedback")
    target_role = st.text_input("Target Role (e.g., Data Scientist, ML Engineer, Data Analyst)")

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


# TAB 3: About

with tab3:
    st.header("ℹ️ About this Project")
    st.markdown("""
    ### AI Resume Analyzer
    This app parses resumes, compares them with job descriptions, and ranks candidates based on:
    - ✅ **Keyword coverage** (must-have and nice-to-have skills)
    - 🤖 **Semantic similarity** using HuggingFace embeddings
    - 📊 **Weighted scoring** with adjustable components

    **Features:**
    - Multi-resume upload and ranking  
    - Progress bars and metrics for clarity  
    - Downloadable CSV of candidate scores  
    - Tailored resume improvement suggestions  

    **Tech Stack:**
    - Python, Streamlit  
    - NLTK, spaCy  
    - HuggingFace Sentence-Transformers  
    - Pandas, scikit-learn  

    ---
    ✨ Built by Harsh Sharma — showcasing NLP + AI for smarter resume analysis.
    """)


# TAB 4: Multi-Role Comparison (auto-detect roles)

with tab4:
    st.header("🔀 Multi-Role Comparison")
    file = st.file_uploader("Upload a single resume (PDF/TXT)", type=["pdf", "txt"], key="multirole")

    if file:
        roles = list_role_names()
        if not roles:
            st.warning("No role profiles found in data/role_profiles/*.yml")
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
                results.append({"Role": r.replace("_", " ").title(), "Score": sc["total_score"], **sc})

            df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
            st.subheader("📊 Role Fit Comparison")
            st.dataframe(df, use_container_width=True)

            best = df.iloc[0]
            st.success(f"✅ Best fit: **{best['Role']}** with score {best['Score']:.2f}")
    else:
        st.info("⬆️ Upload a resume to compare against multiple roles.")
