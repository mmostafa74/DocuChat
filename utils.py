import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.schema import Document
import tempfile
import hashlib
from typing import List


def get_embeddings():
    """Initialize and return HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_documents_hash(documents: List[Document]) -> str:
    """Create a hash from document contents for caching purposes"""
    content_string = ""
    for doc in documents:
        content_string += doc.page_content + str(doc.metadata)
    return hashlib.md5(content_string.encode()).hexdigest()


def load_documents_from_files(uploaded_files) -> List[Document]:
    """Load and process documents from uploaded files"""
    documents = []

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load document based on file type
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == "txt":
                loader = TextLoader(tmp_file_path, encoding="utf-8")
            elif file_extension == "csv":
                loader = CSVLoader(tmp_file_path)
            elif file_extension in ["doc", "docx"]:
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
            else:
                continue

            docs = loader.load()

            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)

        except Exception as e:
            print(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    return documents


def create_vector_store(
    documents: List[Document],
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Create FAISS vector store from documents"""
    if not documents:
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def get_relevant_context(vectorstore, query: str, num_docs: int = 3) -> tuple:
    """Get relevant context and sources from vector store"""
    if not vectorstore:
        return "", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs})
    relevant_docs = retriever.get_relevant_documents(query)

    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    sources = list(
        set([doc.metadata.get("source", "Unknown") for doc in relevant_docs])
    )

    return context, sources


def create_rag_prompt(context: str, question: str) -> str:
    """Create a RAG-enhanced prompt"""
    return f"""Based on the following context from the uploaded documents, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please mention that."""
