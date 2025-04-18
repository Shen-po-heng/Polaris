import gradio as gr
from services.rag_service import RAGService

def create_gradio_interface():
    rag_service = RAGService()

    interface = gr.Interface(
        fn=rag_service.answer_query,
        allow_flagging="never",
        inputs=[
            gr.Files(label="Upload PDF File", file_count="multiple", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Output"),
        title="RAG Chatbot",
        description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    )
    return interface
