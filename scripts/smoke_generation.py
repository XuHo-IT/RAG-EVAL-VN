"""Smoke test generation: đưa context → LLM trả lời + in token count.

Chạy:  python -m scripts.smoke_generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generation import generate


def main():
    context = [
        "Mỗi nhân viên chính thức được hưởng 12 ngày phép có lương trong một năm.",
        "Nhân viên có thâm niên trên 5 năm được cộng thêm 2 ngày phép mỗi năm.",
    ]
    question = "Nhân viên chính thức có bao nhiêu ngày phép năm?"
    out = generate(question, context)
    print("Câu hỏi:", question)
    print("Trả lời:", out["answer"])
    print(f"token_in={out['n_tokens_in']} token_out={out['n_tokens_out']} "
          f"gen_latency={out['gen_latency']:.2f}s")
    assert out["answer"].strip(), "Câu trả lời rỗng!"
    print("OK ✓ generation hoạt động.")


if __name__ == "__main__":
    main()
