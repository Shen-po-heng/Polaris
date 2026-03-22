"""Gradio UI — entry point for the web interface.

Catches domain exceptions from the service layer and converts them
into Gradio errors for user-friendly display.
"""

import gradio as gr

from core.exceptions import PolarisError
from services.rag_service import RAGService
from utils.logger import get_logger

logger = get_logger(__name__)


def create_gradio_interface() -> gr.Interface:
    rag_service = RAGService()

    def answer_with_error_handling(file_objs: list[str], query: str) -> str:
        try:
            return rag_service.answer_query(file_objs, query)
        except PolarisError as exc:
            logger.error("Request failed: %s", exc)
            raise gr.Error(str(exc)) from exc

    return gr.Interface(
        fn=answer_with_error_handling,
        allow_flagging="never",
        inputs=[
            gr.Files(
                label="Upload Documents (PDF / DOCX / TXT / MD)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                type="filepath",
            ),
            gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Ask anything about your documents…",
            ),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Polaris — Local Research Assistant",
        description=(
            "Upload one or more PDF documents and ask any question. "
            "Answers include source citations (file name + page number)."
        ),
    )
