from typing import Dict, List
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .embeddings import embed_texts, cosine_sim

def keyword_counts(text: str, keywords: List[str]) -> int:
    txt = text.lower()
    count = 0
    for kw in keywords:
        # simple exact term match; could be improved with lemmas
        pattern = re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
        count += len(pattern.findall(txt))
    return count

def keyword_coverage(text: str, keywords: List[str]) -> float:
    txt = text.lower()
    hits = 0
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', txt):
            hits += 1
    return hits / max(1, len(keywords))

def keyword_score(text: str, must_have: List[str], nice_to_have: List[str]) -> Dict[str, float]:
    must_cov = keyword_coverage(text, must_have)
    nice_cov = keyword_coverage(text, nice_to_have)
    return {"must_cov": must_cov, "nice_cov": nice_cov}

def tfidf_overlap(resume_text: str, job_text: str) -> float:
    v = TfidfVectorizer(stop_words="english", max_features=5000)
    X = v.fit_transform([job_text, resume_text])
    # cosine similarity between tfidf vectors
    a = X[0].toarray()[0]
    b = X[1].toarray()[0]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

def embedding_match(resume_text: str, job_text: str) -> float:
    embs = embed_texts([job_text, resume_text])
    return float((embs[0] * embs[1]).sum())  # cosine since normalized

def overall_score(resume_text: str, job_text: str, weights: Dict[str, float], must_have: List[str], nice_to_have: List[str]) -> Dict[str, float]:
    ks = keyword_score(resume_text, must_have, nice_to_have)
    emb = embedding_match(resume_text, job_text)
    tfidf = tfidf_overlap(resume_text, job_text)
    # Blend: embeddings + tfidf averaged as "semantic", then keyword with weights
    semantic = 0.5 * emb + 0.5 * tfidf
    total = (
        weights.get("embeddings", 0.4) * semantic +
        weights.get("keyword_must", 0.45) * ks["must_cov"] +
        weights.get("keyword_nice", 0.15) * ks["nice_cov"]
    )
    return {
        "must_coverage": ks["must_cov"],
        "nice_coverage": ks["nice_cov"],
        "embedding_sim": emb,
        "tfidf_sim": tfidf,
        "semantic_sim": semantic,
        "total_score": total
    }
