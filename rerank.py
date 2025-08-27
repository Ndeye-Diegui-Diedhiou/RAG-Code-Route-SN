
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

# Cross-encoder is optional but enabled by default if torch/transformers are available
_HAS_CE = False
_CE_MODEL = None
_CE_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    import torch
    _HAS_CE = True
except Exception:
    _HAS_CE = False

def _ensure_ce_loaded(model_name: str = _CE_NAME):
    global _CE_MODEL, _HAS_CE
    if not _HAS_CE:
        return False
    if _CE_MODEL is None:
        try:
            _CE_MODEL = _CrossEncoder(model_name)
        except Exception:
            _HAS_CE = False
            return False
    return True

def bm25_rerank(query: str, docs: List[str], top_k: int = 5) -> List[int]:
    tokenized_corpus = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    ranked_idxs = np.argsort(scores)[::-1][:top_k]
    return ranked_idxs.tolist()

def crossencoder_rerank(query: str, docs: List[str], model_name: str = _CE_NAME, top_k: int = 5) -> List[int]:
    # Lazy load CE
    ok = _ensure_ce_loaded(model_name)
    if not ok:
        return list(range(min(top_k, len(docs))))
    pairs = [(query, d) for d in docs]
    scores = _CE_MODEL.predict(pairs)
    ranked_idxs = np.argsort(scores)[::-1][:top_k]
    return ranked_idxs.tolist()

def hybrid_indices(query: str, docs: List[str], top_k: int = 5, use_crossencoder: bool = True) -> List[int]:
    # BM25 as primary classical signal (expand candidate pool)
    candidate_k = min(len(docs), max(top_k*3, 10))
    bm25_idx = bm25_rerank(query, docs, top_k=candidate_k)
    chosen_docs = [docs[i] for i in bm25_idx]
    if use_crossencoder and _ensure_ce_loaded():
        ce_idx_local = crossencoder_rerank(query, chosen_docs, top_k=top_k)
        return [bm25_idx[i] for i in ce_idx_local]
    # Otherwise keep top_k from BM25
    return bm25_idx[:top_k]
