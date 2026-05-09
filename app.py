import streamlit as st
import tempfile
import os

# Must happen before rag.py imports so os.getenv() works
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from rag import load_and_chunk, build_vectorstore, build_qa_chain

st.set_page_config(page_title="Financial RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Financial Document Q&A")
st.caption("Ask questions about any financial document — powered by RAG")


with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

    st.divider()
    st.caption("Built with LangChain · FAISS · Groq Llama 3.1")

# Keep chat history and pipeline in session so it survives reruns
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Only reprocess if a new file was uploaded
if uploaded_file and uploaded_file.name != st.session_state.current_file:
    with st.spinner("Reading and indexing document..."):
        # Save to temp file since PyPDFLoader needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks, pages = load_and_chunk(tmp_path)
        vectorstore = build_vectorstore(chunks)
        chain, retriever = build_qa_chain(vectorstore, pages[0].page_content )

        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
        os.unlink(tmp_path)

    st.success(f"Ready! Indexed {len(chunks)} chunks. Ask anything below.")

# Chat interface
if st.session_state.chain:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    # Render previous messages on rerun with sources
                    for src in msg["sources"]:
                        st.caption(f"Page {src['page']}: {src['text']}")

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                docs = st.session_state.retriever.invoke(question)
                sources = [
                    {"page": d.metadata.get("page", "?"),
                     "text": d.page_content[:120] + "..."}
                    for d in docs
                ]
                st.write(answer)
                # Show sources in a collapsible section
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(f"Page {src['page']}: {src['text']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    st.info("Upload a PDF in the sidebar to get started.")