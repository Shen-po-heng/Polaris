from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from utils.logger import get_logger
import gradio as gr
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K
from models.model_manager import ModelManager

logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_manager.initialize_models()

    def process_document(self, file_paths):
        try:
            all_documents = []
            for file_path in file_paths:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path.split("\\")[-1]  # Add filename to metadata
                    doc.metadata["page"] = doc.metadata.get("page", "unknown")
                    # print(f"[DEBUG] Loaded doc chunk meta: {doc.metadata}")
                all_documents.extend(docs)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_documents( all_documents )
            # # **DEBUG**:
            # for i, chunk in enumerate(chunks[:3]):
            #     print(f"[DEBUG] Chunk #{i} meta: {chunk.metadata}")
            vectordb = Chroma.from_documents(chunks, self.model_manager.embedding_model)
            return vectordb.as_retriever(search_kwargs={"k": SEARCH_K})
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise gr.Error("Error processing the document. Please try again.")

    def answer_query(self, file_objs, query):
        try:
            retriever_obj = self.process_document(file_objs)
            qa = RetrievalQA.from_chain_type(
                llm=self.model_manager.llm,
                chain_type="stuff",
                retriever=retriever_obj,
                return_source_documents=True
            )
            response = qa.invoke(query)
            # Extract result text
            result_text = response.get('result', '')
            # Extract helpful answer if present
            helpful_answer_start = result_text.find("Helpful Answer:")
            if helpful_answer_start != -1:
                helpful_answer = result_text[helpful_answer_start + len("Helpful Answer:"):].strip()
            else:
                helpful_answer = result_text.strip()
            print("response:",response)
            # Extract and format source information
            source_docs = response.get('source_documents', [])
            print("[DEBUG] Retrieved source_documents count:", len(source_docs))
            source_info = []
            for doc in source_docs:
                print("[DEBUG] Source doc metadata:", doc.metadata)
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                source_info.append(f"{source} (Page {page})")
            print("Source:",source_info)
            # Format the citation text
            citation_text = "\n\nSources:\n" + "\n".join(sorted(set(source_info))) if source_info else ""
            return helpful_answer + citation_text

        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return f"Error: {str(e)}"