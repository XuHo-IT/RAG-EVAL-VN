"""Định nghĩa các cấu hình RAG mẫu — CẤU HÌNH LÀ DỮ LIỆU, KHÔNG PHẢI CODE.

Thêm cấu hình mới = thêm 1 dict vào CONFIGS, không cần sửa code lõi (blueprint 01 §5).
Mỗi cấu hình mô tả đầy đủ một pipeline: chunk → embed → retrieve (± rerank).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagConfig:
    id: str
    chunk_strategy: str = "fixed"          # fixed | recursive | semantic
    chunk_size: int = 256
    chunk_overlap: int = 32
    embedding_model: str = "bge-m3"        # bge-m3 | e5
    top_k: int = 5
    use_rerank: bool = False
    rerank_top_n: int = 3

    def label(self) -> str:
        rr = f"rerank giữ {self.rerank_top_n}" if self.use_rerank else "không rerank"
        return (f"{self.id}: {self.chunk_strategy}/{self.chunk_size} · "
                f"{self.embedding_model} · top {self.top_k} · {rr}")


# Bộ cấu hình mẫu để so sánh (giống tinh thần bảng blueprint 03 §3)
CONFIGS: list[RagConfig] = [
    RagConfig(id="A", chunk_strategy="fixed", chunk_size=256,
              embedding_model="bge-m3", top_k=5, use_rerank=False),
    RagConfig(id="B", chunk_strategy="fixed", chunk_size=512,
              embedding_model="e5", top_k=5, use_rerank=False),
    RagConfig(id="C", chunk_strategy="semantic", chunk_size=256,
              embedding_model="bge-m3", top_k=8, use_rerank=True, rerank_top_n=3),
    RagConfig(id="D", chunk_strategy="recursive", chunk_size=256,
              embedding_model="bge-m3", top_k=5, use_rerank=False),
]

CONFIGS_BY_ID = {c.id: c for c in CONFIGS}


def get_config(config_id: str) -> RagConfig:
    return CONFIGS_BY_ID[config_id]
