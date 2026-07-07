"""Bước 1 — Chunking: cắt tài liệu dài thành đoạn nhỏ.

Đây là bước quyết định chất lượng RAG nhiều nhất. Hỗ trợ 3 chiến lược để so sánh:
- fixed:     cắt theo số token cố định (nhanh, hay cắt giữa câu).
- recursive: cắt theo phân cấp đoạn → câu → từ, giữ ranh giới tự nhiên.
- semantic:  cắt theo điểm chuyển ý (dựa embedding câu), giữ ngữ nghĩa tốt nhất.

Đặc thù tiếng Việt (blueprint 02 §1):
- Đếm token bằng CHÍNH tokenizer của embedding model (token VN ≠ token EN).
- Tách câu bằng dấu câu VN, tránh cắt nhầm ở "TP.HCM", số thập phân...
- Ưu tiên underthesea nếu cài được, có fallback regex.
"""

from __future__ import annotations

import re

import numpy as np

from models import get_embedder

# Ranh giới phân cấp cho recursive splitting (từ thô → mịn)
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]


# ── Đếm token theo tokenizer của embedding model ─────────────────────────────
def _get_tokenizer(model_name: str):
    return get_embedder(model_name).tokenizer


def count_tokens(text: str, model_name: str) -> int:
    tok = _get_tokenizer(model_name)
    return len(tok.encode(text, add_special_tokens=False))


# ── Tách câu tiếng Việt ──────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành câu, ưu tiên underthesea, fallback regex VN-aware."""
    text = text.strip()
    if not text:
        return []
    try:
        from underthesea import sent_tokenize

        sents = sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        return _regex_split_sentences(text)


# Viết tắt thường gặp không phải hết câu (bảo vệ khỏi bị cắt nhầm)
_ABBREV = r"(?:TP|Tp|Q|P|Đ|ĐT|TS|ThS|GS|PGS|BS|KS|Mr|Mrs|Ms|Dr|vs|v\.v)"


def _regex_split_sentences(text: str) -> list[str]:
    # Tạm che dấu chấm trong viết tắt và số thập phân
    protected = re.sub(rf"\b{_ABBREV}\.", lambda m: m.group(0).replace(".", "␟"), text)
    protected = re.sub(r"(\d)\.(\d)", r"\1␟\2", protected)
    # Cắt sau . ! ? ; theo sau bởi khoảng trắng
    parts = re.split(r"(?<=[.!?;])\s+", protected)
    return [p.replace("␟", ".").strip() for p in parts if p.strip()]


# ── Chiến lược 1: fixed-size ─────────────────────────────────────────────────
def chunk_fixed(
    text: str, model_name: str, size: int = 256, overlap: int = 32
) -> list[str]:
    """Cắt theo số token cố định, có phần chồng lấn (overlap) giữ ngữ cảnh."""
    tok = _get_tokenizer(model_name)
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(ids), step):
        window = ids[i : i + size]
        chunk = tok.decode(window, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if i + size >= len(ids):
            break
    return chunks


# ── Chiến lược 2: recursive ──────────────────────────────────────────────────
def chunk_recursive(
    text: str, model_name: str, size: int = 256, overlap: int = 32
) -> list[str]:
    """Cắt theo phân cấp ranh giới tự nhiên, gộp lại tới ~size token/đoạn."""
    pieces = _recursive_split(text, _RECURSIVE_SEPARATORS, model_name, size)
    return _merge_pieces(pieces, model_name, size, overlap)


def _recursive_split(
    text: str, separators: list[str], model_name: str, size: int
) -> list[str]:
    if count_tokens(text, model_name) <= size or not separators:
        return [text] if text.strip() else []
    sep, rest = separators[0], separators[1:]
    out: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if count_tokens(part, model_name) <= size:
            out.append(part)
        else:
            out.extend(_recursive_split(part, rest, model_name, size))
    return out


def _merge_pieces(
    pieces: list[str], model_name: str, size: int, overlap: int
) -> list[str]:
    """Gộp các mẩu nhỏ liền nhau thành chunk gần size token, có overlap theo mẩu."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for p in pieces:
        p_tok = count_tokens(p, model_name)
        if cur and cur_tok + p_tok > size:
            chunks.append(" ".join(cur))
            # giữ lại phần đuôi làm overlap
            keep, keep_tok = [], 0
            for q in reversed(cur):
                qt = count_tokens(q, model_name)
                if keep_tok + qt > overlap:
                    break
                keep.insert(0, q)
                keep_tok += qt
            cur, cur_tok = keep, keep_tok
        cur.append(p)
        cur_tok += p_tok
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ── Chiến lược 3: semantic ───────────────────────────────────────────────────
def chunk_semantic(
    text: str,
    model_name: str,
    size: int = 256,
    overlap: int = 0,
    breakpoint_percentile: int = 25,
) -> list[str]:
    """Cắt tại điểm chuyển ý: gộp các câu liên tiếp khi còn tương đồng cao,
    ngắt khi độ tương đồng giữa 2 câu liền kề tụt xuống dưới ngưỡng (percentile).
    Vẫn tôn trọng trần `size` token để không tạo chunk quá dài."""
    from rag.embedding import embed_texts

    sents = split_sentences(text)
    if len(sents) <= 1:
        return sents

    vecs = embed_texts(sents, model_name)  # đã normalize
    # cosine giữa câu i và i+1 = tích vô hướng (vì đã normalize)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = np.percentile(sims, breakpoint_percentile)

    chunks: list[str] = []
    cur = [sents[0]]
    cur_tok = count_tokens(sents[0], model_name)
    for i in range(1, len(sents)):
        s_tok = count_tokens(sents[i], model_name)
        shift = sims[i - 1] < threshold  # điểm chuyển ý
        too_big = cur_tok + s_tok > size
        if shift or too_big:
            chunks.append(" ".join(cur))
            cur, cur_tok = [sents[i]], s_tok
        else:
            cur.append(sents[i])
            cur_tok += s_tok
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ── Dispatcher theo cấu hình ─────────────────────────────────────────────────
_STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
}


def chunk_document(
    text: str,
    strategy: str,
    model_name: str,
    size: int = 256,
    overlap: int = 32,
) -> list[str]:
    """Cắt tài liệu theo chiến lược chỉ định. `model_name` dùng để đếm token
    (và để embed câu nếu strategy = semantic)."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"Chiến lược chunk '{strategy}' không hợp lệ. "
                         f"Chọn: {list(_STRATEGIES)}")
    fn = _STRATEGIES[strategy]
    return fn(text, model_name, size=size, overlap=overlap)
