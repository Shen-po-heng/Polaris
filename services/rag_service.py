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
                    doc.metadata["source"] = file_path.split("/")[-1]  # Add filename to metadata
                    # doc.metadata["page"] = doc.metadata.get("page", "unknown")
                all_documents.extend(docs)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_documents( all_documents )
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
                return_source_documents=False
            )
            response = qa.invoke(query)
            print(response)
            # Extract the helpful answer after "Question:" and "Helpful Answer:"
            result_text = response.get('result', '')
            
            # Assuming "Helpful Answer:" appears in the response
            helpful_answer_start = result_text.find("Helpful Answer:")
            
            if helpful_answer_start != -1:
                # Extract the text after "Helpful Answer:"
                helpful_answer = result_text[helpful_answer_start + len("Helpful Answer:"):].strip()
                return helpful_answer
            else:
                return "No helpful answer found."

        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return f"Error: {str(e)}"
