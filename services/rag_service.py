"""RAG service — document processing and question answering.

Responsibilities:
  - Load and chunk PDF documents.
  - Build an in-memory vector store from chunks.
  - Answer user queries with source citations.

Note: the in-memory Chroma instance is rebuilt on every query.
      Persistent ChromaDB caching is addressed in Phase 2.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K
from core.exceptions import DocumentLoadError, IndexingError, QueryError
from models.model_manager import ModelManager
from utils.logger import get_logger
from utils.security import validate_file

logger = get_logger(__name__)


class RAGService:
    def __init__(self) -> None:
        self.model_manager = ModelManager()
        self.model_manager.initialize_models()

    def process_document(self, file_paths: list[str]) -> object:
        """Load PDFs, chunk text and build a retriever.

        Args:
            file_paths: List of absolute paths to PDF files.

        Returns:
            A LangChain retriever ready for similarity search.

        Raises:
            IndexingError: If loading or indexing fails.
        """
        try:
            all_documents = []
            for raw_path in file_paths:
                validated = validate_file(raw_path)
                loader = PyPDFLoader(str(validated))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = validated.name  # cross-platform filename
                    doc.metadata["page"] = doc.metadata.get("page", "unknown")
                all_documents.extend(docs)
                logger.info("Loaded %d pages from '%s'", len(docs), validated.name)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_documents(all_documents)
            logger.debug("Split into %d chunks across %d documents", len(chunks), len(file_paths))

            vectordb = Chroma.from_documents(chunks, self.model_manager.embedding_model)
            return vectordb.as_retriever(search_kwargs={"k": SEARCH_K})

        except (DocumentLoadError, IndexingError):
            raise
        except Exception as exc:
            logger.exception("Error processing documents")
            raise IndexingError("Error processing the document. Please try again.") from exc

    def answer_query(self, file_objs: list[str], query: str) -> str:
        """Answer a query using RAG over the provided documents.

        Args:
            file_objs: List of PDF file paths (from Gradio file upload).
            query: The user's question.

        Returns:
            Answer text with source citations appended.

        Raises:
            QueryError: If the answer step fails unexpectedly.
        """
        try:
            retriever = self.process_document(file_objs)
            qa = RetrievalQA.from_chain_type(
                llm=self.model_manager.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            response = qa.invoke(query)
            logger.debug("RAG response: %s", response)

            result_text: str = response.get("result", "")

            # Some prompt templates prefix the answer with "Helpful Answer:"
            marker = "Helpful Answer:"
            idx = result_text.find(marker)
            answer = result_text[idx + len(marker):].strip() if idx != -1 else result_text.strip()

            source_docs = response.get("source_documents", [])
            logger.debug("Retrieved %d source documents", len(source_docs))

            source_info: list[str] = []
            for doc in source_docs:
                logger.debug("Source metadata: %s", doc.metadata)
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                source_info.append(f"{source} (Page {page})")

            logger.debug("Sources: %s", source_info)

            citation_text = (
                "\n\nSources:\n" + "\n".join(sorted(set(source_info)))
                if source_info
                else ""
            )
            return answer + citation_text

        except (IndexingError, DocumentLoadError):
            raise
        except Exception as exc:
            logger.exception("Error answering query")
            raise QueryError(str(exc)) from exc
