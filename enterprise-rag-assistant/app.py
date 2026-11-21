"""Streamlit interface for the Enterprise RAG Assistant."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import answer_question, build_vector_store, load_pdf_documents

load_dotenv()

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise RAG Assistant")
st.caption("Ask questions about your PDFs and get answers grounded in retrieved passages.")

with st.sidebar:
    st.header("Configuration")
    st.write("**LLM:**", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    st.write("**Embeddings:**", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    st.info("Your PDF text is sent to the configured model API for embedding and answer generation.")
    if st.button("Clear document index", use_container_width=True):
        st.session_state.pop("vector_store", None)
        st.session_state.pop("indexed_files", None)
        st.session_state.pop("chunk_count", None)
        st.session_state.pop("messages", None)
        st.rerun()

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Add your OPENAI_API_KEY to a .env file to enable document indexing and answers.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    current_names = st.session_state.get("indexed_files", [])

    if st.button("Process documents", type="primary", disabled=not os.getenv("OPENAI_API_KEY")):
        try:
            with st.spinner("Reading PDFs, creating embeddings, and building the FAISS index..."):
                documents = load_pdf_documents(uploaded_files)
                vector_store, chunk_count = build_vector_store(documents)
                st.session_state["vector_store"] = vector_store
                st.session_state["indexed_files"] = file_names
                st.session_state["chunk_count"] = chunk_count
                st.session_state["messages"] = []
            st.success(f"Indexed {len(file_names)} PDF(s) into {chunk_count} searchable chunks.")
        except Exception as exc:
            st.error(f"Could not process the documents: {exc}")

if st.session_state.get("vector_store") is not None:
    st.success(
        f"Ready: {len(st.session_state.get('indexed_files', []))} document(s), "
        f"{st.session_state.get('chunk_count', 0)} chunks indexed."
    )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['file']}**, page **{source['page']}**"
                        )

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        st.session_state.setdefault("messages", []).append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    answer, sources = answer_question(
                        st.session_state["vector_store"], question
                    )
                    source_data = [
                        {
                            "file": doc.metadata.get("source", "Unknown file"),
                            "page": doc.metadata.get("page", "Unknown"),
                        }
                        for doc in sources
                    ]
                    st.markdown(answer)
                    if source_data:
                        with st.expander("Retrieved sources"):
                            for source in source_data:
                                st.markdown(
                                    f"- **{source['file']}**, page **{source['page']}**"
                                )
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_data,
                        }
                    )
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
else:
    st.markdown(
        """
        ### How it works
        1. Upload PDFs.
        2. Extract and split the text into overlapping chunks.
        3. Convert chunks into embeddings and store them in FAISS.
        4. Retrieve relevant passages for your question.
        5. Generate a grounded answer with source references.

        **Tip:** Start with a small PDF to keep the first embedding request quick.
        """
    )
