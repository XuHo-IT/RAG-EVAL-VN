"""Bước 3 — Retrieval: tìm các chunk liên quan nhất với câu hỏi.

Cơ bản (vector search):
  1. Embed câu hỏi bằng cùng model đã embed chunk.
  2. Tìm top-k chunk gần nhất trong FAISS (cosine qua IndexFlatIP, vector đã normalize).

Nâng cao (rerank, tùy chọn — blueprint 02 §3):
  Sau khi vector search lấy nhiều chunk, dùng bge-reranker chấm lại độ liên quan
  thật sự với câu hỏi rồi giữ rerank_top_n chunk tốt nhất.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np

from models import get_reranker
from rag.embedding import embed_query


@dataclass
class VectorIndex:
    """Chỉ mục FAISS trong RAM cho một (tài liệu × chunking × embedding)."""

    index: faiss.Index
    chunks: list[str]
    model_name: str


def build_index(chunks: list[str], embeddings: np.ndarray, model_name: str) -> VectorIndex:
    """Dựng FAISS IndexFlatIP từ embedding đã normalize (inner product = cosine)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return VectorIndex(index=index, chunks=chunks, model_name=model_name)


def search(
    vindex: VectorIndex,
    question: str,
    top_k: int = 5,
    *,
    use_rerank: bool = False,
    rerank_top_n: int = 3,
    rerank_pool: int = 20,
) -> dict:
    """Tìm chunk liên quan. Nếu use_rerank: lấy rộng (rerank_pool) rồi rerank
    và giữ rerank_top_n tốt nhất.

    Returns dict:
        chunks:          list[str] chunk cuối cùng
        scores:          list[float] điểm tương ứng
        retrieval_latency: giây
    """
    t0 = time.perf_counter()

    qvec = embed_query(question, vindex.model_name).reshape(1, -1)
    fetch_k = max(top_k, rerank_pool) if use_rerank else top_k
    fetch_k = min(fetch_k, len(vindex.chunks))

    scores, idxs = vindex.index.search(qvec, fetch_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    cand = [(vindex.chunks[i], s) for i, s in zip(idxs, scores) if i != -1]

    if use_rerank and cand:
        cand = _rerank(question, [c for c, _ in cand], rerank_top_n)
    else:
        cand = cand[:top_k]

    latency = time.perf_counter() - t0
    return {
        "chunks": [c for c, _ in cand],
        "scores": [float(s) for _, s in cand],
        "retrieval_latency": latency,
    }


def _rerank(question: str, chunks: list[str], top_n: int) -> list[tuple[str, float]]:
    reranker = get_reranker()
    pairs = [[question, c] for c in chunks]
    raw = np.asarray(reranker.predict(pairs), dtype="float32")
    # bge-reranker xuất logit → sigmoid đưa về [0, 1] để hiển thị dễ đọc
    scores = 1.0 / (1.0 + np.exp(-raw))
    ranked = sorted(zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
