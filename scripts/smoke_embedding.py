"""Smoke test embedding: kiểm tra nạp model + cosine 2 câu VN gần nghĩa cao.

Chạy:  python -m scripts.smoke_embedding   (từ thư mục gốc dự án)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rag.embedding import embed_texts


def main():
    model = "bge-m3"
    texts = [
        "Chính sách nghỉ phép của công ty như thế nào?",
        "Quy định về ngày phép năm cho nhân viên ra sao?",
        "Hôm nay trời mưa rất to ở Hà Nội.",
    ]
    vecs = embed_texts(texts, model)
    print(f"[{model}] shape = {vecs.shape}")

    sim_related = float(np.dot(vecs[0], vecs[1]))
    sim_unrelated = float(np.dot(vecs[0], vecs[2]))
    print(f"cosine(gần nghĩa)   = {sim_related:.3f}")
    print(f"cosine(khác chủ đề) = {sim_unrelated:.3f}")
    assert sim_related > sim_unrelated, "Cặp gần nghĩa phải có cosine cao hơn!"
    print("OK ✓ embedding hoạt động đúng.")


if __name__ == "__main__":
    main()
