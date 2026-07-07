"""Orchestrator lõi — chạy 1 hoặc nhiều cấu hình RAG trên cùng (tài liệu, câu hỏi).

Điểm tối ưu quan trọng: cache theo (nội dung tài liệu × chunking × embedding).
Nhiều cấu hình khác nhau ở top_k/rerank nhưng chung cách chunk + embedding model
sẽ TÁI DÙNG cùng một FAISS index thay vì chunk + embed lại (blueprint 01 §5).

Tách khỏi app.py để test độc lập (smoke_pipeline.py) và để UI chỉ lo hiển thị.
"""

from __future__ import annotations

import hashlib

from config import RagConfig
from eval.metrics import evaluate_run
from rag.chunking import chunk_document
from rag.embedding import embed_texts
from rag.generation import generate
from rag.retrieval import build_index, search


def _index_key(text: str, cfg: RagConfig) -> tuple:
    doc_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return (doc_hash, cfg.chunk_strategy, cfg.chunk_size, cfg.chunk_overlap,
            cfg.embedding_model)


def run_config(
    text: str,
    question: str,
    cfg: RagConfig,
    reference_answer: str | None = None,
    *,
    index_cache: dict | None = None,
    do_generate: bool = True,
) -> dict:
    """Chạy trọn pipeline cho 1 cấu hình. Trả về dict gồm chunks lấy ra,
    câu trả lời, và metrics."""
    cache = index_cache if index_cache is not None else {}

    # 1-2. Chunk + embed + dựng index (có cache)
    key = _index_key(text, cfg)
    if key not in cache:
        chunks = chunk_document(
            text, cfg.chunk_strategy, cfg.embedding_model,
            size=cfg.chunk_size, overlap=cfg.chunk_overlap,
        )
        embeddings = embed_texts(chunks, cfg.embedding_model)
        cache[key] = build_index(chunks, embeddings, cfg.embedding_model)
    vindex = cache[key]

    # 3. Retrieval (± rerank)
    ret = search(
        vindex, question, top_k=cfg.top_k,
        use_rerank=cfg.use_rerank, rerank_top_n=cfg.rerank_top_n,
    )

    # 4. Generation
    if do_generate:
        gen = generate(question, ret["chunks"])
    else:
        gen = {"answer": "(bỏ qua generation)", "n_tokens_in": 0,
               "n_tokens_out": 0, "gen_latency": 0.0}

    run = {
        "config_id": cfg.id,
        "n_chunks_total": len(vindex.chunks),
        "retrieved_chunks": ret["chunks"],
        "retrieval_scores": ret["scores"],
        "retrieval_latency": ret["retrieval_latency"],
        **gen,
    }
    run["metrics"] = evaluate_run(run, reference_answer, score_answer=do_generate)
    return run


def run_many(
    text: str,
    question: str,
    configs: list[RagConfig],
    reference_answer: str | None = None,
    *,
    do_generate: bool = True,
    progress=None,
) -> list[dict]:
    """Chạy tuần tự nhiều cấu hình, tái dùng index chung. `progress` là callable
    tùy chọn (vd gr.Progress) nhận (fraction, desc)."""
    cache: dict = {}
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        if progress:
            progress(i / n, f"Đang chạy cấu hình {cfg.id} ...")
        results.append(
            run_config(text, question, cfg, reference_answer,
                       index_cache=cache, do_generate=do_generate)
        )
    if progress:
        progress(1.0, "Xong")
    return results
