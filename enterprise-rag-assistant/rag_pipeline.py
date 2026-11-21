"""PDF ingestion, vector indexing, retrieval, and grounded answer generation."""

from __future__ import annotations

import os
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(uploaded_files: Iterable) -> list[Document]:
    """Load uploaded PDF files and preserve filename/page metadata."""
    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        temp_path = None
        try:
            import tempfile

            suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded_file.getvalue())
                temp_path = temp.name

            pages = PyPDFLoader(temp_path).load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                # PyPDFLoader pages are zero-indexed; show human-friendly page numbers.
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
            documents.extend(pages)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return documents


def build_vector_store(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[FAISS, int]:
    """Split documents and create a FAISS index using OpenAI embeddings."""
    if not documents:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded PDFs did not produce any text chunks.")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def answer_question(
    vector_store: FAISS,
    question: str,
    k: int = 4,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and generate an answer constrained to their content."""
    if not question.strip():
        raise ValueError("Please enter a question.")

    matches = vector_store.similarity_search(question, k=k)
    if not matches:
        return "I could not find relevant passages in the uploaded documents.", []

    context_parts = []
    for index, doc in enumerate(matches, start=1):
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(
            f"[Source {index}: {source}, page {page}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful document question-answering assistant. "
                "Answer using only the supplied context. If the context does not "
                "contain the answer, say you cannot find the answer in the uploaded "
                "documents. Do not invent facts. Cite supporting passages inline "
                "using [Source 1], [Source 2], etc.",
            ),
            (
                "human",
                "Question: {question}\n\nRetrieved context:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    response = (prompt | model).invoke(
        {"question": question, "context": context}
    )
    return str(response.content), matches
