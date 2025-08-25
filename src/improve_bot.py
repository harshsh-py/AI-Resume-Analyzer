from typing import Dict, List

TIPS = {
    "python": "Show Python depth: projects, repos, or Kaggle notebooks. Mention libraries (pandas, numpy, scikit-learn).",
    "sql": "Quantify SQL work: window funcs, CTEs, query optimization, and real datasets you queried.",
    "machine learning": "List 2–3 models you've trained end-to-end. Include metrics (ROC-AUC, F1) and business impact.",
    "statistics": "Add statistical chops: hypothesis testing, confidence intervals, A/B tests you ran.",
    "pandas": "Highlight data wrangling: groupby, merges, time-series resampling.",
    "numpy": "Mention vectorization and performance improvements from NumPy usage.",
    "scikit-learn": "Name pipelines, cross-validation, and model selection techniques you used.",
    "data visualization": "Include 1–2 plots with links or images; mention Matplotlib/Plotly/Seaborn.",
    "feature engineering": "Describe domain features you created and why they helped.",
    "model evaluation": "Report metrics clearly; compare baselines vs. final models.",
    "deep learning": "If relevant, note PyTorch/TensorFlow projects with problem, data size, and results.",
    "nlp": "If NLP, state tasks (classification, NER, QA) and datasets. Link to demos.",
    "mlops": "Mention experiment tracking (MLflow), CI, model registry, and deployment.",
    "cloud": "Show pipelines on AWS/GCP/Azure; name services used.",
}

def gap_analysis(resume_text: str, must_have: List[str], nice_to_have: List[str]) -> Dict[str, List[str]]:
    lower = resume_text.lower()
    missing_must = [k for k in must_have if k.lower() not in lower]
    missing_nice = [k for k in nice_to_have if k.lower() not in lower]
    return {"missing_must": missing_must, "missing_nice": missing_nice}

def suggestions_from_gaps(gaps: Dict[str, List[str]]) -> List[str]:
    out = []
    for bucket in ["missing_must", "missing_nice"]:
        for kw in gaps.get(bucket, []):
            tip = TIPS.get(kw.lower(), f"Add concrete evidence of {kw} (projects, metrics, or links).")
            out.append(f"- {kw.title()}: {tip}")
    if not out:
        out.append("- Strong coverage already. Consider tightening bullets, quantifying impact, and linking to work.")
    return out
def suggest_improvements(resume_text: str, target_role: str):
    """
    Very simple resume improvement helper.
    Looks at the role name and suggests missing skills/phrases.
    """
    suggestions = []

    role_keywords = {
        "data scientist": ["python", "machine learning", "deep learning", "sql", "statistics"],
        "ml engineer": ["mlops", "docker", "kubernetes", "model deployment", "pytorch", "tensorflow"],
        "data analyst": ["sql", "excel", "tableau", "power bi", "data visualization"]
    }

    # normalize
    role = target_role.lower()
    if role in role_keywords:
        for kw in role_keywords[role]:
            if kw not in resume_text.lower():
                suggestions.append(f"Consider adding experience with **{kw}** to strengthen your {target_role} resume.")

    if not suggestions:
        suggestions.append(f"Your resume already looks good for a {target_role} role. 👍")

    return suggestions
