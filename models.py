"""Điểm nạp model DUY NHẤT của toàn app.

Nguyên tắc (blueprint 05 §5 - "Cache model"): nạp mỗi model đúng 1 lần rồi
tái dùng qua mọi request. Nạp lazy — chỉ nạp khi có cấu hình thực sự cần tới.

Device:
- Embedding: mặc định CPU; đặt env EMBED_DEVICE=cuda để chạy trên GPU
  (mỗi lúc chỉ nên có 1 embedding model trên GPU với VRAM nhỏ).
- Reranker + LLM: luôn CPU cho ổn định và khớp môi trường HF Spaces free.
"""

from __future__ import annotations

import os
import sys
import threading

# Console Windows mặc định cp1252 → ép UTF-8 để log tiếng Việt không crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Registry: tên ngắn dùng trong config → model id trên HuggingFace Hub ──────
EMBEDDING_MODELS: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-base",
}
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # "cpu" | "cuda"

# ── Cache nội bộ (nạp 1 lần) ─────────────────────────────────────────────────
_lock = threading.Lock()
_embedders: dict[str, object] = {}
_reranker = None
_llm = None  # (tokenizer, model)


def get_embedder(name: str):
    """Trả về SentenceTransformer cho tên model ngắn ('bge-m3' | 'e5')."""
    if name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Embedding model '{name}' chưa hỗ trợ. "
            f"Chọn một trong: {list(EMBEDDING_MODELS)}"
        )
    with _lock:
        if name not in _embedders:
            from sentence_transformers import SentenceTransformer

            print(f"[models] Nạp embedding '{name}' ({EMBEDDING_MODELS[name]}) "
                  f"trên {EMBED_DEVICE} ...")
            _embedders[name] = SentenceTransformer(
                EMBEDDING_MODELS[name], device=EMBED_DEVICE
            )
        return _embedders[name]


def get_reranker():
    """Trả về CrossEncoder (bge-reranker-v2-m3), nạp lazy.

    Dùng sentence-transformers CrossEncoder thay cho FlagEmbedding.FlagReranker
    vì FlagReranker (1.4) gọi tokenizer.prepare_for_model đã bị bỏ ở transformers 5.x.
    Cùng model, cùng chất lượng, nhưng tương thích stack hiện tại.
    """
    global _reranker
    with _lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            print(f"[models] Nạp reranker ({RERANKER_MODEL}) ...")
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")
        return _reranker


def get_llm():
    """Trả về (tokenizer, model) của Qwen2.5-1.5B-Instruct trên CPU, nạp lazy."""
    global _llm
    with _lock:
        if _llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[models] Nạp LLM ({LLM_MODEL}) trên CPU ...")
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                dtype=torch.float32,  # CPU dùng fp32
                device_map="cpu",
            )
            _llm = (tokenizer, model)
        return _llm
