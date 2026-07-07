"""Bước 5 — Evaluation: đo lường & so sánh các cấu hình (blueprint 03).

4 nhóm chỉ số:
- Retrieval Precision: các chunk lấy ra có thật sự liên quan đáp án chuẩn không.
- Answer Relevance:    câu trả lời cuối có khớp đáp án chuẩn không.
- Latency:             thời gian retrieval + generation.
- Cost:                token input + output của LLM.

Lưu ý trung thực (blueprint 03 §5): precision/relevance là ĐO TỰ ĐỘNG GẦN ĐÚNG
dựa trên tương đồng ngữ nghĩa, và chỉ tính được khi có đáp án chuẩn.
"""

from __future__ import annotations

import numpy as np

from rag.embedding import embed_texts

# Embedding model dùng RIÊNG cho việc chấm điểm, cố định để so sánh công bằng
# giữa các cấu hình (không phụ thuộc embedding model của từng cấu hình).
JUDGE_MODEL = "bge-m3"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # a, b đã normalize (embed_texts normalize sẵn) → cosine = dot product
    return float(np.dot(a, b))


def retrieval_precision(retrieved_chunks: list[str], reference_answer: str) -> float:
    """Trung bình cosine giữa mỗi chunk lấy ra và đáp án chuẩn.
    Cao = các chunk lấy ra sát nội dung đáp án (retrieval trúng đích)."""
    if not retrieved_chunks or not reference_answer.strip():
        return 0.0
    ref = embed_texts([reference_answer], JUDGE_MODEL)[0]
    chunk_vecs = embed_texts(retrieved_chunks, JUDGE_MODEL)
    sims = chunk_vecs @ ref
    return float(np.mean(sims))


def answer_relevance(answer: str, reference_answer: str) -> float:
    """Cosine giữa câu trả lời sinh ra và đáp án chuẩn."""
    if not answer.strip() or not reference_answer.strip():
        return 0.0
    vecs = embed_texts([answer, reference_answer], JUDGE_MODEL)
    return _cosine(vecs[0], vecs[1])


def evaluate_run(
    run: dict, reference_answer: str | None, *, score_answer: bool = True
) -> dict:
    """Tính metrics cho 1 lần chạy cấu hình.

    Args:
        run: kết quả pipeline, cần các khóa:
             retrieved_chunks, answer, retrieval_latency, gen_latency,
             n_tokens_in, n_tokens_out
        reference_answer: đáp án chuẩn (có thể None → bỏ precision/relevance).
        score_answer: False khi bỏ qua generation → relevance = None
                      (không chấm câu trả lời không tồn tại).

    Returns dict metrics (giá trị None nếu không tính được).
    """
    has_ref = bool(reference_answer and reference_answer.strip())
    total_latency = run.get("retrieval_latency", 0.0) + run.get("gen_latency", 0.0)
    cost = run.get("n_tokens_in", 0) + run.get("n_tokens_out", 0)

    return {
        "precision": (
            round(retrieval_precision(run["retrieved_chunks"], reference_answer), 4)
            if has_ref else None
        ),
        "relevance": (
            round(answer_relevance(run["answer"], reference_answer), 4)
            if has_ref and score_answer else None
        ),
        "retrieval_latency": round(run.get("retrieval_latency", 0.0), 3),
        "gen_latency": round(run.get("gen_latency", 0.0), 3),
        "total_latency": round(total_latency, 3),
        "cost_tokens": cost,
    }


# Hướng "thắng" của mỗi metric: True = cao hơn tốt hơn
METRIC_HIGHER_BETTER = {
    "precision": True,
    "relevance": True,
    "total_latency": False,
    "cost_tokens": False,
}


def find_winners(rows: list[dict]) -> dict[str, str]:
    """Với mỗi metric, trả về config id thắng (để highlight ở UI).
    `rows` là list dict, mỗi dict có 'config_id' + các metric."""
    winners: dict[str, str] = {}
    for metric, higher in METRIC_HIGHER_BETTER.items():
        best_id, best_val = None, None
        for r in rows:
            v = r.get(metric)
            if v is None:
                continue
            if best_val is None or (v > best_val if higher else v < best_val):
                best_val, best_id = v, r["config_id"]
        if best_id is not None:
            winners[metric] = best_id
    return winners
