# Financial Document RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload any financial PDF — annual reports, 10-Ks, earnings statements — and ask questions in plain English.

## Demo
Upload a PDF → ask questions → get cited answers with page numbers.

## Tech Stack
- LangChain — RAG pipeline
- FAISS — vector similarity search
- Google Gemini — embeddings
- Groq Llama 3.1 — LLM
- Streamlit — UI

## How to Run

```bash
git clone https://github.com/MilapPrajapati70/financial-rag-chatbot
cd financial-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:
```
GOOGLE_API_KEY=your-key
GROQ_API_KEY=your-key
```

```bash
streamlit run app.py
```

## Features
- Upload any PDF document
- Semantic search over document chunks
- Cited answers with page numbers
- Conversation history
- Works with large financial documents
