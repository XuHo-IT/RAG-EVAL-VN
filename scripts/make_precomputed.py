"""Sinh dữ liệu demo tức thì: chạy pipeline thật 1 lần rồi lưu TOÀN BỘ kết quả
(bao gồm chunk lấy ra) vào data/precomputed_demo.json.

App dùng file này cho nút "Xem kết quả mẫu (tức thì)" — người dùng thấy ngay bảng
so sánh mà không phải đợi tải model + chạy 40s/cấu hình trên CPU.

Chạy lại khi đổi sample/cấu hình:  python -m scripts.make_precomputed
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIGS, get_config
from pipeline import run_many
from utils import read_document

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_vn.txt"
OUT = ROOT / "data" / "precomputed_demo.json"

QUESTION = "Nhân viên chính thức được bao nhiêu ngày phép năm và điều kiện cộng thêm?"
REFERENCE = ("Nhân viên chính thức được 12 ngày phép có lương mỗi năm; thâm niên "
             "trên 5 năm được cộng thêm 2 ngày. Phép chưa dùng chuyển tối đa 5 ngày "
             "sang quý một năm sau.")


def main():
    text = read_document(str(SAMPLE))
    results = run_many(text, QUESTION, list(CONFIGS), REFERENCE, do_generate=True)

    # Chỉ giữ các trường app cần để render (bỏ object nặng)
    slim = []
    for r in results:
        slim.append({
            "config_id": r["config_id"],
            "config_label": get_config(r["config_id"]).label(),
            "n_chunks_total": r["n_chunks_total"],
            "retrieved_chunks": r["retrieved_chunks"],
            "retrieval_scores": r["retrieval_scores"],
            "answer": r["answer"],
            "metrics": r["metrics"],
        })

    payload = {
        "document": text,
        "question": QUESTION,
        "reference": REFERENCE,
        "results": slim,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu {OUT.relative_to(ROOT)} ({len(slim)} cấu hình).")


if __name__ == "__main__":
    main()
