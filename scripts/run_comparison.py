"""Chạy so sánh đầy đủ 4 cấu hình A/B/C/D trên tài liệu mẫu → in bảng metrics.

Dùng để: verify nhánh e5 + reranker, và lấy SỐ LIỆU THẬT điền README.
Chạy:  python -m scripts.run_comparison
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIGS, get_config
from eval.metrics import find_winners
from pipeline import run_many
from utils import read_document

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_vn.txt"


def main():
    text = read_document(str(SAMPLE))
    question = "Nhân viên chính thức được bao nhiêu ngày phép năm và điều kiện cộng thêm?"
    reference = ("Nhân viên chính thức được 12 ngày phép có lương mỗi năm; "
                 "thâm niên trên 5 năm được cộng thêm 2 ngày. Phép chưa dùng "
                 "chuyển tối đa 5 ngày sang quý một năm sau.")

    print(f"Câu hỏi: {question}\n")
    results = run_many(text, question, list(CONFIGS), reference, do_generate=True)

    flat = [{"config_id": r["config_id"], **r["metrics"]} for r in results]
    winners = find_winners(flat)

    # Bảng
    print(f"{'Cấu hình':<42} {'Prec':>6} {'Relev':>6} {'Lat(s)':>7} {'Cost':>6}")
    print("-" * 72)
    for r in results:
        m = r["metrics"]
        cfg = get_config(r["config_id"])
        print(f"{cfg.label():<42} {m['precision']:>6} {m['relevance']:>6} "
              f"{m['total_latency']:>7} {m['cost_tokens']:>6}")
    print("\nThắng mỗi cột:")
    for metric, cid in winners.items():
        print(f"  {metric:16s}: cấu hình {cid}")

    print("\nCâu trả lời:")
    for r in results:
        print(f"  [{r['config_id']}] {r['answer'][:140]}")

    # Xuất JSON để tiện đọc lại
    out = Path(__file__).resolve().parents[1] / "comparison_result.json"
    payload = [{"config": get_config(r["config_id"]).label(),
                "answer": r["answer"], **r["metrics"]} for r in results]
    out.write_text(json.dumps({"question": question, "reference": reference,
                               "results": payload, "winners": winners},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu chi tiết vào {out.name}")


if __name__ == "__main__":
    main()
