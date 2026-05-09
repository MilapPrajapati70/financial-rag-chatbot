import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


# Load the PDF and break it into smaller pieces

def load_and_chunk(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No PDF found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Print first 500 chars so we can see what was actually extracted
    print("\n--- RAW TEXT PREVIEW ---")
    print(pages[0].page_content[:500])
    print("------------------------\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,      # increased overlap
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks")
    return chunks,pages

# Store chunk embeddings in FAISS for fast lookup

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
        
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Connect retriever + LLM into a single pipeline

PROMPT_TEMPLATE = """You are a document assistant. 
Answer the question using only the context provided below.
If you can't find the answer, say so honestly.
Reference page numbers where relevant.

Context:
{context}

Question: {question}

Answer:"""
def build_qa_chain(vectorstore, first_page_text=""):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

      # Use same embedding model for queries to ensure consistency
    query_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4},
        embedding=query_embeddings
        
    )
   # Format retrieved chunks with page numbers and some of the first page for extra context
    def format_docs(docs):
        base = f"[Always available - Page 0 header]\n{first_page_text[:300]}\n\n"
        retrieved = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        return base + retrieved

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# Quick terminal test — run directly to verify pipeline works

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/sample.pdf"

    chunks = load_and_chunk(pdf_path)
    vectorstore = build_vectorstore(chunks)
    chain, retriever = build_qa_chain(vectorstore, pages[0].page_content)

    print("\nRAG chain ready. Ask a question:")
    question = input("> ")

    answer = chain.invoke(question)
    print("\nAnswer:", answer)

    print("\nSources:")
    for doc in retriever.invoke(question):
        print(f"  - Page {doc.metadata.get('page', '?')}: {doc.page_content[:80]}...")