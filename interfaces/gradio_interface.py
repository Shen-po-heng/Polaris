"""Gradio UI — Phase 2.5 two-tab interface.

Tab 1 💬 Chat
  - Upload new files + index
  - Select indexed sources for querying
  - Multi-turn chat with persistent history

Tab 2 📚 Library
  - View all indexed sources
  - Select and delete from ChromaDB
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from core.exceptions import PolarisError
from services.chat_history import ChatHistoryManager
from services.rag_service import RAGService
from services.summarizer import Summarizer, _MAX_CHARS, _SUMMARY_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_HISTORY_TURNS = 5

css = """
body, .gradio-container { font-size: 16px !important; }
.message { font-size: 15px !important; line-height: 1.6 !important; }
textarea, input[type="text"] { font-size: 15px !important; }
button { font-size: 15px !important; }
details summary { font-size: 15px !important; font-weight: 600; cursor: pointer; }
details p { font-size: 14px !important; line-height: 1.7 !important; margin: 8px 0; }
h2 { font-size: 20px !important; }
.status-ok  { color: #16a34a; font-weight: 600; }
.status-err { color: #dc2626; font-weight: 600; }
"""


def create_gradio_interface() -> gr.Blocks:
    rag = RAGService()
    summarizer = Summarizer()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _paths(file_objs) -> list[str]:
        if not file_objs:
            return []
        if isinstance(file_objs[0], str):
            return file_objs
        return [f.name for f in file_objs]

    def _all_sources() -> list[str]:
        return rag.list_sources()

    # ── Tab 1 callbacks ───────────────────────────────────────────────────────

    def on_index(file_objs):
        """Index uploaded files, refresh both source selectors."""
        paths = _paths(file_objs)
        if not paths:
            return (
                gr.update(),          # chat source selector
                gr.update(),          # library source selector
                "⚠️ No files selected.",
            )
        try:
            names = rag.index_documents(paths)
            sources = _all_sources()
            msg = f"✅ Indexed: {', '.join(names)}"
            return (
                gr.update(choices=sources, value=sources),  # auto-select all
                gr.update(choices=sources, value=[]),
                msg,
            )
        except PolarisError as exc:
            return gr.update(), gr.update(), f"❌ {exc}"

    def on_chat(query: str, history: list, selected: list):
        if not query.strip():
            return history, history, ""
        sources = selected or _all_sources()
        if not sources:
            new = history + [[query, "No documents in library. Please upload and index first."]]
            return new, new, ""
        try:
            trimmed = history[-_MAX_HISTORY_TURNS:]
            answer = rag.query(query, history=trimmed, selected_sources=sources)
        except PolarisError as exc:
            logger.error("Chat error: %s", exc)
            answer = f"Error: {exc}"

        new_history = history + [[query, answer]]
        ChatHistoryManager.save(new_history)
        return new_history, new_history, ""

    def on_new_chat():
        ChatHistoryManager.clear()
        return [], [], ""

    def on_summarise(selected: list, file_objs):
        """Summarise selected sources.

        Priority:
        1. For each selected source, if the user uploaded a matching file — use
           DocumentLoader (original text, highest quality).
        2. Otherwise pull stored chunks from ChromaDB (no re-upload needed).
        """
        sources_to_summarise = list(selected or [])
        paths = _paths(file_objs)

        if not sources_to_summarise and not paths:
            return gr.update(value="<p>Select sources or upload files first.</p>", visible=True)

        # Build a map: filename → upload path (for freshly uploaded files)
        upload_map = {Path(p).name: p for p in paths}

        # If nothing selected but files uploaded, summarise uploaded files
        if not sources_to_summarise:
            sources_to_summarise = list(upload_map.keys())

        html_parts = []
        for src in sources_to_summarise:
            try:
                if src in upload_map:
                    # Use original file — best quality
                    result = summarizer.summarise([upload_map[src]])
                    summary = result.get(src, "[No summary returned]")
                else:
                    # Pull from ChromaDB chunks
                    text = rag.get_source_text(src)
                    if not text:
                        summary = "[No stored text found — please re-upload.]"
                    else:
                        prompt = _SUMMARY_PROMPT.format(
                            filename=src, text=text[:_MAX_CHARS]
                        )
                        summary = summarizer._llm.chat(prompt)
            except PolarisError as exc:
                summary = f"[Error: {exc}]"

            html_parts.append(
                f"<details open><summary><b>{src}</b></summary>"
                f"<p>{summary}</p></details>"
            )

        return gr.update(value="\n".join(html_parts), visible=True)

    # ── Tab 2 callbacks ───────────────────────────────────────────────────────

    def on_refresh_library():
        sources = _all_sources()
        count = len(sources)
        return (
            gr.update(choices=sources, value=[]),
            f"**{count} source(s) in library.**",
        )

    def on_delete(selected: list):
        if not selected:
            return gr.update(), "⚠️ Nothing selected."
        try:
            n = rag.delete_sources(selected)
            sources = _all_sources()
            return (
                gr.update(choices=sources, value=[]),
                f"✅ Deleted {len(selected)} source(s) ({n} chunks).",
            )
        except PolarisError as exc:
            return gr.update(), f"❌ {exc}"

    # ── Layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="Polaris — Chat with Your Documents", css=css) as demo:
        gr.Markdown(
            "# 📚 Polaris — Chat with Your Documents / 與文件對話\n"
            "> Local · Private · No API key needed for Ollama"
        )

        with gr.Tabs():
            # ── Tab 1: Chat ───────────────────────────────────────────────────
            with gr.Tab("💬 Chat / 對話"):
                with gr.Row(equal_height=False):

                    # Left panel
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 📂 Upload & Index / 上傳建立索引")
                        file_upload = gr.Files(
                            label="PDF / DOCX / TXT / MD",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            type="filepath",
                        )
                        index_btn = gr.Button("📥 Index / 建立索引", variant="primary")
                        index_status = gr.Markdown("")

                        gr.Markdown("### 🔍 Select Sources / 選擇引用來源")
                        source_selector = gr.CheckboxGroup(
                            choices=_all_sources(),
                            value=_all_sources(),
                            label="Sources to query (empty = all)",
                        )

                        summarise_btn = gr.Button("📋 Summarize / 摘要", variant="secondary")
                        summary_html = gr.HTML(value="", visible=False)

                    # Right panel
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown("### 💬 Conversation / 對話")
                        chatbot = gr.Chatbot(
                            label="",
                            height=450,
                            show_copy_button=True,
                            value=ChatHistoryManager.load(),
                        )
                        chat_state = gr.State(ChatHistoryManager.load())

                        with gr.Row():
                            query_box = gr.Textbox(
                                label="",
                                placeholder="Ask a question… / 輸入問題…",
                                lines=2,
                                scale=5,
                                show_label=False,
                            )
                            send_btn = gr.Button("➤ Send / 送出", variant="primary", scale=1)

                        new_chat_btn = gr.Button("🗑 New Chat / 新對話", variant="stop", size="sm")

            # ── Tab 2: Library ────────────────────────────────────────────────
            with gr.Tab("📚 Library / 文件庫"):
                with gr.Row():
                    refresh_lib_btn = gr.Button("🔄 Refresh / 重新整理", variant="secondary")
                    lib_status = gr.Markdown("Click Refresh to load library.")

                lib_selector = gr.CheckboxGroup(
                    choices=_all_sources(),
                    value=[],
                    label="Indexed sources — select to delete / 勾選後刪除",
                )
                delete_btn = gr.Button("🗑 Delete Selected / 刪除所選", variant="stop")

        # ── Wire events ───────────────────────────────────────────────────────

        index_btn.click(
            fn=on_index,
            inputs=[file_upload],
            outputs=[source_selector, lib_selector, index_status],
        )

        send_btn.click(
            fn=on_chat,
            inputs=[query_box, chat_state, source_selector],
            outputs=[chatbot, chat_state, query_box],
        )
        query_box.submit(
            fn=on_chat,
            inputs=[query_box, chat_state, source_selector],
            outputs=[chatbot, chat_state, query_box],
        )

        new_chat_btn.click(
            fn=on_new_chat,
            inputs=[],
            outputs=[chatbot, chat_state, query_box],
        )

        summarise_btn.click(
            fn=on_summarise,
            inputs=[source_selector, file_upload],
            outputs=[summary_html],
        )

        refresh_lib_btn.click(
            fn=on_refresh_library,
            inputs=[],
            outputs=[lib_selector, lib_status],
        )

        delete_btn.click(
            fn=on_delete,
            inputs=[lib_selector],
            outputs=[lib_selector, lib_status],
        )

        # Sync library selector when tab is loaded
        demo.load(
            fn=lambda: gr.update(choices=_all_sources(), value=_all_sources()),
            inputs=[],
            outputs=[source_selector],
        )

    return demo
