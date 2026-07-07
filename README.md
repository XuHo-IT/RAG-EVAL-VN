---
title: RAG Eval Playground VN
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# 🔍 RAG Eval Playground VN

> Công cụ **đo lường thay vì đoán mò** khi chọn cấu hình RAG cho tài liệu tiếng Việt.

Upload tài liệu tiếng Việt, đặt câu hỏi, rồi chạy **song song nhiều cấu hình RAG**
(chunking / embedding / rerank / top-k) và xem **bảng so sánh metrics** để biết
cấu hình nào cho câu trả lời tốt nhất — tất cả bằng open model, miễn phí.

## Vấn đề

Xây một hệ thống RAG buộc phải chọn: cắt tài liệu (chunk) to hay nhỏ? Dùng embedding
model nào? Có cần rerank không? Lấy bao nhiêu đoạn (top-k)? Hầu hết chọn bừa rồi hy
vọng. Với **tiếng Việt** — vốn không tách từ bằng khoảng trắng — cấu hình tối ưu cho
tiếng Anh chưa chắc đúng.

## Giải pháp

Playground này chạy nhiều cấu hình trên cùng tài liệu + câu hỏi rồi so sánh bằng số
liệu khách quan: **retrieval precision, answer relevance, latency, cost (token)**.

## Tính năng

- ⚖️ **Đa cấu hình song song** — so sánh chunking (fixed / recursive / semantic),
  embedding (bge-m3 vs multilingual-e5), bật/tắt rerank, đổi top-k.
- 📊 **Bảng metrics + biểu đồ** — highlight 🏆 cấu hình thắng mỗi cột.
- 🇻🇳 **Tiếng Việt là công dân hạng nhất** — tách câu VN (underthesea, có fallback),
  đếm token bằng chính tokenizer của embedding model.
- 🆓 **Miễn phí hoàn toàn** — open model, không API key, chạy local hoặc HF Spaces.

## Cách chạy (local)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate       |  Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt

python app.py            # mở http://127.0.0.1:7860
```

> Lần chạy đầu tải model (~6–7GB: bge-m3, e5, bge-reranker, Qwen2.5-1.5B) nên mất
> vài phút. Sau đó model được cache lại. Mặc định chạy CPU; đặt `EMBED_DEVICE=cuda`
> để chạy embedding trên GPU.

Kiểm thử nhanh từng bước (không cần UI):

```bash
python -m scripts.smoke_embedding     # kiểm tra embedding
python -m scripts.smoke_generation    # kiểm tra LLM
python -m scripts.smoke_pipeline      # chạy trọn pipeline 1 cấu hình
```

## Kết quả rút ra ⭐

Chạy thật 4 cấu hình trên tài liệu nội quy nhân sự VN mẫu, câu hỏi *"Nhân viên chính
thức được bao nhiêu ngày phép năm và điều kiện cộng thêm?"* (CPU, Qwen2.5-1.5B):

| Cấu hình | Precision | Relevance | Latency | Cost (token) |
|---|---|---|---|---|
| A · fixed 256 · bge-m3 · top5 | 0.709 | 0.927 | 37.3s | 1689 |
| B · fixed 512 · **e5** · top5 | 0.681 | 0.912 | 55.6s | 1849 |
| C · semantic · bge-m3 · **+rerank→3** | **0.760** 🏆 | **0.943** 🏆 | **37.0s** 🏆 | **540** 🏆 |
| D · recursive 256 · bge-m3 · top5 | 0.719 | 0.937 | 43.1s | 1301 |

**Ba insight thật:**

1. **Rerank vừa tăng chất lượng vừa GIẢM chi phí — điều phản trực giác.** Cấu hình C
   (semantic + rerank giữ 3 chunk) đạt precision cao nhất (0.76) *và* tốn ít token nhất
   (540 vs 1689 của A — **giảm ~68%**). Lý do: rerank lọc bỏ chunk nhiễu, chỉ đưa 3 đoạn
   đúng nhất vào LLM thay vì 5 đoạn thô → prompt ngắn hơn → vừa rẻ vừa sinh nhanh hơn.

2. **Rerank có "chi phí ẩn" ở retrieval, không phải ở tổng thời gian.** Riêng bước
   retrieval của C mất 14.2s (vs 0.16s của A) vì cross-encoder chấm lại 8 chunk trên CPU.
   Nhưng **tổng** latency C (37.0s) vẫn ngang A (37.3s) vì phần generation nhanh hơn hẳn
   (ít token). Bài học: đo latency phải tách retrieval vs generation mới thấy đúng bức tranh.

3. **Chunk 512 + e5 (B) thua chunk 256 + bge-m3 (A) trên tài liệu Việt này.** B có
   precision thấp hơn (0.68 < 0.71), tốn token hơn và **chậm nhất (55.6s)** do chunk to
   đẩy nhiều token vào LLM. Khẳng định: "chunk to hơn" và "đổi embedding model" không
   mặc nhiên tốt hơn — phải đo. bge-m3 nhỉnh hơn e5 cho văn bản hành chính VN ở đây.

> **Trung thực về phép đo**: đây là 1 tài liệu + 1 câu hỏi, mang tính minh họa, không
> phải kết luận thống kê. Giá trị của playground là cho bạn *tự đo trên tài liệu của
> mình* rồi quyết định — chứ không phải áp đặt một cấu hình "tốt nhất" cố định.

## Kiến trúc

```
app.py            # Gradio UI + trình bày kết quả
pipeline.py       # orchestrator: chạy nhiều cấu hình, cache index dùng chung
config.py         # các cấu hình mẫu (cấu hình là dữ liệu, không phải code)
models.py         # nạp & cache model (embedder / reranker / LLM) 1 lần
rag/              # chunking · embedding · retrieval · generation
eval/metrics.py   # precision · relevance · latency · cost
data/sample_vn.txt
scripts/          # smoke test độc lập từng bước
```

## Tech stack

Python · Gradio · sentence-transformers (bge-m3, multilingual-e5, và
bge-reranker-v2-m3 qua CrossEncoder) · FAISS · Transformers (Qwen2.5-1.5B-Instruct) ·
pypdf · underthesea.

## Lưu ý về tính trung thực

Precision & relevance là **đo tự động gần đúng** dựa trên tương đồng ngữ nghĩa và chỉ
tính được khi có đáp án chuẩn — không phải điểm tuyệt đối. Mục tiêu là *so sánh tương
đối giữa các cấu hình*, không phải chấm điểm chính xác 100%.
