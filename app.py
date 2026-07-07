"""RAG Eval Playground VN — Gradio UI + điều phối hiển thị.

Luồng: người dùng nạp tài liệu (hoặc dùng sample) → nhập câu hỏi + đáp án chuẩn →
chọn các cấu hình → chạy → xem bảng metrics + biểu đồ + câu trả lời cạnh nhau.

Logic RAG nằm ở pipeline.py; file này chỉ lo giao diện và trình bày kết quả.
"""

from __future__ import annotations

import os
import sys

# Console Windows mặc định cp1252 → ép UTF-8 để log tiếng Việt không crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import gradio as gr
import pandas as pd

from config import CONFIGS, get_config
from eval.metrics import find_winners
from pipeline import run_many
from utils import read_document

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_vn.txt")

CONFIG_CHOICES = [(c.label(), c.id) for c in CONFIGS]
DEFAULT_SELECTED = [c.id for c in CONFIGS[:3]]

# ── Thiết kế: tối giản nền ĐEN – chữ TRẮNG sáng, không icon/emoji ─────────────
# Font dùng system stack (không tải font ngoài) để giữ app tự chứa và chạy offline.
CUSTOM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#0B0B0B",
    body_text_color="#FFFFFF",
    body_text_color_subdued="#C9C9C9",
    background_fill_primary="#0B0B0B",
    background_fill_secondary="#141414",
    block_background_fill="#111111",
    block_border_color="#2A2A2A",
    block_border_width="1px",
    block_label_text_color="#FFFFFF",
    block_title_text_color="#FFFFFF",
    block_radius="12px",
    input_background_fill="#161616",
    input_border_color="#2A2A2A",
    input_border_color_focus="#FFFFFF",
    input_placeholder_color="#8A8A8A",
    button_large_radius="6px",
    button_small_radius="6px",
    button_primary_background_fill="#FFFFFF",
    button_primary_background_fill_hover="#DDDDDD",
    button_primary_text_color="#111111",
    button_primary_border_color="#FFFFFF",
    button_secondary_background_fill="#1A1A1A",
    button_secondary_background_fill_hover="#242424",
    button_secondary_border_color="#2A2A2A",
    button_secondary_text_color="#FFFFFF",
)

CUSTOM_CSS = """
:root, .gradio-container {
  --ink:#FFFFFF; --muted:#C9C9C9; --line:#2A2A2A; --bg:#0B0B0B;
  --font: 'Helvetica Neue', Arial, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
}
body, .gradio-container { background: var(--bg) !important; color: #FFFFFF !important; }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.block, .form, .gr-box, .gr-panel { box-shadow: none !important; }
/* Ép mọi phần chữ sáng trắng, không để xám tối (trừ nút primary nền trắng) */
.gradio-container label, .gradio-container p, .gradio-container span,
.gradio-container li, .gradio-container td, .gradio-container th,
.gradio-container .prose, .gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container .gr-check-radio label { color: #FFFFFF !important; }
button.primary, .primary button, button[variant="primary"] { color: #111111 !important; }
#app-header { border-bottom: 1px solid var(--line); padding: 4px 0 20px; margin-bottom: 4px; }
#app-title { font-family: Georgia, 'Times New Roman', serif; font-size: 34px;
  line-height: 1.1; letter-spacing: -0.02em; color: #FFFFFF !important; margin: 0; font-weight: 600; }
#app-sub { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
  letter-spacing: 0.08em; text-transform: uppercase; color: #C9C9C9 !important; margin-top: 12px; }
#app-note { font-size: 13px; line-height: 1.6; color: #C9C9C9 !important; margin-top: 10px;
  max-width: 640px; }
.section-label { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
  letter-spacing: 0.12em; text-transform: uppercase; color: #C9C9C9 !important;
  border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 6px 0 4px; }
table td, table th { border-color: var(--line) !important; }
@media (max-width: 820px) {
  #main-row { flex-direction: column !important; }
  #app-title { font-size: 26px; }
  .gradio-container { padding: 12px !important; }
}
"""


# ── Xử lý sự kiện ────────────────────────────────────────────────────────────
def load_sample() -> str:
    try:
        return read_document(SAMPLE_PATH)
    except Exception as e:  # pragma: no cover
        return f"(Không nạp được sample: {e})"


def on_upload(file) -> str:
    if file is None:
        return ""
    try:
        return read_document(file.name)
    except Exception as e:
        raise gr.Error(f"Lỗi đọc tài liệu: {e}")


def run_comparison(document_text, question, reference, selected_ids, do_generate,
                   progress=gr.Progress()):
    # Kiểm tra đầu vào thân thiện
    if not document_text or not document_text.strip():
        raise gr.Error("Chưa có nội dung tài liệu. Hãy nạp sample hoặc upload file.")
    if not question or not question.strip():
        raise gr.Error("Hãy nhập câu hỏi.")
    if not selected_ids:
        raise gr.Error("Hãy chọn ít nhất một cấu hình để so sánh.")

    configs = [get_config(cid) for cid in selected_ids]
    ref = reference.strip() if reference else None

    results = run_many(
        document_text, question, configs, ref,
        do_generate=do_generate, progress=progress,
    )

    metrics_df = _build_metrics_table(results)
    answers_md = _build_answers_md(results)
    chunks_md = _build_chunks_md(results)
    chart_df = _build_chart_df(results)
    return metrics_df, chart_df, answers_md, chunks_md


# ── Dựng bảng / chart / markdown ─────────────────────────────────────────────
_METRIC_LABELS = {
    "config": "Cấu hình",
    "precision": "Precision (cao tốt)",
    "relevance": "Relevance (cao tốt)",
    "total_latency": "Latency giây (thấp tốt)",
    "cost_tokens": "Cost token (thấp tốt)",
}


def _build_metrics_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    flat = []
    for r in results:
        m = r["metrics"]
        flat.append({"config_id": r["config_id"], **m})
    winners = find_winners(flat)

    for r in results:
        m = r["metrics"]
        cid = r["config_id"]
        cfg = get_config(cid)

        def cell(metric):
            v = m.get(metric)
            if v is None:
                return "-"
            mark = "  BEST" if winners.get(metric) == cid else ""
            return f"{v}{mark}"

        rows.append({
            _METRIC_LABELS["config"]: cfg.label(),
            _METRIC_LABELS["precision"]: cell("precision"),
            _METRIC_LABELS["relevance"]: cell("relevance"),
            _METRIC_LABELS["total_latency"]: cell("total_latency"),
            _METRIC_LABELS["cost_tokens"]: cell("cost_tokens"),
        })
    return pd.DataFrame(rows)


def _build_chart_df(results: list[dict]) -> pd.DataFrame:
    """Dạng long để vẽ grouped bar: precision & relevance theo cấu hình."""
    rows = []
    for r in results:
        m = r["metrics"]
        for metric in ("precision", "relevance"):
            v = m.get(metric)
            if v is not None:
                rows.append({"config": r["config_id"], "metric": metric, "value": v})
    if not rows:  # không có đáp án chuẩn → không vẽ được chất lượng
        rows.append({"config": "—", "metric": "precision", "value": 0.0})
    return pd.DataFrame(rows)


def _build_answers_md(results: list[dict]) -> str:
    parts = ["## Câu trả lời theo từng cấu hình\n"]
    for r in results:
        cfg = get_config(r["config_id"])
        parts.append(f"### {cfg.label()}\n{r['answer']}\n")
    return "\n".join(parts)


def _build_chunks_md(results: list[dict]) -> str:
    parts = ["## Các đoạn (chunk) được lấy ra\n"]
    for r in results:
        cfg = get_config(r["config_id"])
        parts.append(f"### {cfg.label()}  "
                     f"*(tổng {r['n_chunks_total']} chunk)*")
        for i, (c, s) in enumerate(zip(r["retrieved_chunks"], r["retrieval_scores"])):
            snippet = c[:300] + ("…" if len(c) > 300 else "")
            parts.append(f"- **#{i + 1}** (score {s:.3f}): {snippet}")
        parts.append("")
    return "\n".join(parts)


# ── Giao diện (tối giản đen–trắng, không icon/emoji) ─────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Eval Playground VN", theme=CUSTOM_THEME,
                   css=CUSTOM_CSS) as demo:
        with gr.Column(elem_id="app-header"):
            gr.HTML(
                "<h1 id='app-title'>RAG Eval Playground VN</h1>"
                "<div id='app-sub'>Đo lường thay vì đoán mò — so sánh cấu hình "
                "RAG cho tài liệu tiếng Việt</div>"
                "<div id='app-note'>Lần chạy đầu cần tải model (1–2 phút). Trên CPU "
                "miễn phí nên chạy 1–2 cấu hình mỗi lần cho mượt; bỏ chọn sinh câu "
                "trả lời để đo nhanh chỉ phần retrieval.</div>"
            )

        with gr.Row(elem_id="main-row", equal_height=False):
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>Tài liệu và câu hỏi</div>")
                document_text = gr.Textbox(
                    label="Nội dung tài liệu", lines=10,
                    placeholder="Nạp tài liệu mẫu hoặc tải tệp lên, nội dung hiển thị ở đây",
                )
                with gr.Row():
                    sample_btn = gr.Button("Dùng tài liệu mẫu", variant="secondary")
                    upload = gr.File(label="Hoặc tải tệp (.pdf, .txt, .md)",
                                     file_types=[".pdf", ".txt", ".md"])
                question = gr.Textbox(label="Câu hỏi", lines=2,
                                      placeholder="Ví dụ: Chính sách nghỉ phép thế nào")
                reference = gr.Textbox(
                    label="Đáp án chuẩn — để chấm precision và relevance, có thể bỏ trống",
                    lines=2,
                )
                configs_sel = gr.CheckboxGroup(
                    choices=CONFIG_CHOICES, value=DEFAULT_SELECTED,
                    label="Cấu hình muốn so sánh",
                )
                do_generate = gr.Checkbox(
                    value=True,
                    label="Sinh câu trả lời bằng LLM — bỏ chọn để chỉ đo retrieval, nhanh hơn",
                )
                run_btn = gr.Button("Chạy so sánh", variant="primary")

            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>Kết quả đo lường</div>")
                gr.Markdown(
                    "Nhãn BEST đánh dấu cấu hình thắng ở mỗi cột. Precision và "
                    "relevance là đo tự động gần đúng, cần có đáp án chuẩn."
                )
                metrics_out = gr.Dataframe(interactive=False, wrap=True)
                chart_out = gr.BarPlot(
                    x="config", y="value", color="metric",
                    color_map={"precision": "#FFFFFF", "relevance": "#888888"},
                    title="Chất lượng theo cấu hình (cao hơn tốt hơn)",
                    x_title="Cấu hình", y_title="Điểm", height=280,
                )

        with gr.Accordion("Câu trả lời từng cấu hình", open=True):
            answers_out = gr.Markdown()
        with gr.Accordion("Các đoạn được lấy ra (xem retrieval hoạt động)", open=False):
            chunks_out = gr.Markdown()

        # Sự kiện
        sample_btn.click(load_sample, outputs=document_text)
        upload.upload(on_upload, inputs=upload, outputs=document_text)
        run_btn.click(
            run_comparison,
            inputs=[document_text, question, reference, configs_sel, do_generate],
            outputs=[metrics_out, chart_out, answers_out, chunks_out],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
