"""Smoke test pipeline: chạy trọn 1 cấu hình trên sample_vn.txt.

Chạy:  python -m scripts.smoke_pipeline
       python -m scripts.smoke_pipeline --no-gen   (bỏ generation cho nhanh)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config
from pipeline import run_config
from utils import read_document

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_vn.txt"


def main():
    do_gen = "--no-gen" not in sys.argv
    text = read_document(str(SAMPLE))
    question = "Nhân viên chính thức được bao nhiêu ngày phép năm?"
    reference = ("Nhân viên chính thức được 12 ngày phép có lương mỗi năm, "
                 "thâm niên trên 5 năm được cộng thêm 2 ngày.")

    cfg = get_config("A")  # fixed/256 · bge-m3 · top5 · no-rerank
    print(f"Cấu hình: {cfg.label()}")
    run = run_config(text, question, cfg, reference, do_generate=do_gen)

    print(f"\nTổng chunk: {run['n_chunks_total']}")
    print(f"Chunk lấy ra ({len(run['retrieved_chunks'])}):")
    for i, (c, s) in enumerate(zip(run["retrieved_chunks"], run["retrieval_scores"])):
        print(f"  #{i+1} (score {s:.3f}): {c[:100]}…")
    print(f"\nCâu trả lời: {run['answer']}")
    print(f"\nMetrics: {run['metrics']}")
    print("\nOK ✓ pipeline chạy thông.")


if __name__ == "__main__":
    main()
