from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        # Small, fast, decent quality
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=True)
    return np.array(embs)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b are 1D vectors
    num = float((a * b).sum())
    return num  # already normalized if using normalize_embeddings=True
