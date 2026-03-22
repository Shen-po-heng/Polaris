from langchain_community.document_loaders import PyPDFLoader
import pprint

file_path = "./testPDF02.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print(f"Document metadata: {docs[0].metadata}")
# print(f"Document content: {docs[0].page_content}")
print(f"Document source: {docs[0].metadata['source']}")
print("Document content preview:")
pprint.pprint(
    docs[0].page_content[:1000]
)  # Print the first 1000 characters of the content
