"""Bước 2 — Embedding: biến văn bản thành vector (đã normalize) để tìm theo nghĩa.

Hỗ trợ 2 model đa ngôn ngữ, mạnh tiếng Việt:
- bge-m3: dùng trực tiếp, không cần prefix.
- e5 (multilingual-e5): CHUẨN yêu cầu prefix 'query:' cho câu hỏi và
  'passage:' cho đoạn văn — bỏ prefix sẽ giảm chất lượng đáng kể.

Vector luôn được normalize (chuẩn L2) để cosine similarity = tích vô hướng,
dùng thẳng với FAISS IndexFlatIP ở bước retrieval.
"""

from __future__ import annotations

import numpy as np

from models import get_embedder


def _apply_e5_prefix(texts: list[str], is_query: bool) -> list[str]:
    prefix = "query: " if is_query else "passage: "
    return [prefix + t for t in texts]


def embed_texts(
    texts: list[str],
    model_name: str,
    *,
    is_query: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Sinh embedding cho danh sách văn bản.

    Args:
        texts: các đoạn/câu cần embed.
        model_name: 'bge-m3' | 'e5'.
        is_query: True nếu đây là câu hỏi (ảnh hưởng prefix của e5).
        batch_size: số văn bản encode mỗi lượt.

    Returns:
        Mảng shape (n, dim), dtype float32, đã normalize L2.
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedder(model_name)

    inputs = texts
    if model_name == "e5":
        inputs = _apply_e5_prefix(texts, is_query)

    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,  # normalize để cosine = dot product
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


def embed_query(text: str, model_name: str) -> np.ndarray:
    """Tiện ích: embed 1 câu hỏi, trả về vector 1 chiều (dim,)."""
    return embed_texts([text], model_name, is_query=True)[0]
