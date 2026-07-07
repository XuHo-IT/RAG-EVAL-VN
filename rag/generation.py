"""Bước 4 — Generation: ghép các chunk vào prompt, gọi LLM sinh câu trả lời.

Nguyên tắc (blueprint 02 §4):
- Prompt yêu cầu model CHỈ dùng thông tin trong các đoạn được đưa (chống bịa).
- Ghi lại số token input/output để tính cost.
- Đo thời gian gọi để tính latency generation.
"""

from __future__ import annotations

import time

from models import get_llm

SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu. "
    "CHỈ dùng thông tin trong các đoạn được cung cấp. "
    "Nếu các đoạn không chứa thông tin để trả lời, hãy nói rõ: "
    '"Tôi không tìm thấy thông tin trong tài liệu." '
    "Trả lời ngắn gọn, bằng tiếng Việt."
)


def _build_user_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(
        f"[Đoạn {i + 1}] {c}" for i, c in enumerate(context_chunks)
    )
    return (
        f"Dựa vào các đoạn sau, trả lời câu hỏi.\n\n"
        f"{context}\n\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời:"
    )


def generate(
    question: str,
    context_chunks: list[str],
    *,
    max_new_tokens: int = 256,
) -> dict:
    """Sinh câu trả lời từ câu hỏi + các chunk ngữ cảnh.

    Returns dict:
        answer:        câu trả lời (str)
        n_tokens_in:   số token prompt đưa vào LLM
        n_tokens_out:  số token sinh ra
        gen_latency:   thời gian sinh (giây)
    """
    tokenizer, model = get_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, context_chunks)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_tokens_in = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic để so sánh cấu hình công bằng
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_latency = time.perf_counter() - t0

    # Chỉ lấy phần token mới sinh (bỏ prompt)
    new_ids = output_ids[0][n_tokens_in:]
    n_tokens_out = int(new_ids.shape[0])
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "n_tokens_in": n_tokens_in,
        "n_tokens_out": n_tokens_out,
        "gen_latency": gen_latency,
    }
