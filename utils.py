"""Tiện ích đọc tài liệu đầu vào (PDF / TXT)."""

from __future__ import annotations

import os


def read_document(path: str) -> str:
    """Trích text từ file .pdf hoặc .txt. Ném lỗi rõ ràng nếu không đọc được."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Không tìm thấy file tài liệu.")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Định dạng '{ext}' chưa hỗ trợ. Dùng .pdf, .txt hoặc .md.")


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(
            "Không trích được text từ PDF (có thể là PDF scan ảnh, chưa OCR)."
        )
    return text
