# 📄 PDF Q&A ChatBot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-12D9A6.svg)](https://langchain.com/)
[![VectorDB](https://img.shields.io/badge/vector_db-FAISS-0052CC.svg)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-Qwen_2.5_72B-8C4FFF.svg)](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)

An interactive, responsive **RAG (Retrieval-Augmented Generation)** web application that allows users to upload PDF documents and have intelligent, context-aware conversations about their content. Powered by **LangChain**, **Streamlit**, **FAISS** (vector store), and **Qwen-2.5-72B** (via HuggingFace Endpoints).

---

## 📸 Application Preview

Below is a preview of the ChatBot answering questions and referencing precise source snippets from the document:

![Application Interface](Screenshot%202026-06-05%20at%2011.24.58.png)

---

## 🔄 RAG Pipeline

The application is powered by a two-phase pipeline — **Ingestion** (run once per PDF) and **Retrieval** (run on every question).

```mermaid
flowchart TD
    subgraph INGESTION["⬆️  INGESTION  —  run once per PDF"]
        A([🗂️ Upload PDF]) --> B[PyPDFLoader\nload pages]
        B --> C[RecursiveCharacterTextSplitter\nchunk_size=1000 · overlap=200]
        C --> D[HuggingFace Embeddings\nall-MiniLM-L6-v2]
        D --> E[(FAISS\nVector Store)]
    end

    subgraph RETRIEVAL["🔍  RETRIEVAL  —  run on every question"]
        F([💬 User Question]) --> G[Embed query\nsame model]
        G --> H{Similarity Search\ntop-k chunks}
        E -- stored\nvectors --> H
        H --> I[create_stuff_documents_chain\nstuff context into prompt]
        I --> J[Qwen-2.5-72B-Instruct\nvia HuggingFace Endpoint]
        J --> K([✅ Answer + Source Citations])
    end

    K --> L[(Session\nChat History)]
    L -. next question .-> F

    style INGESTION fill:#1e3a5f,color:#e0f0ff,stroke:#4a90d9
    style RETRIEVAL fill:#1a3a2a,color:#e0ffe8,stroke:#4ad98a
    style E fill:#0d2b4a,color:#7ec8ff,stroke:#4a90d9
    style L fill:#0d2b4a,color:#7ec8ff,stroke:#4a90d9
    style A fill:#2e6da4,color:#fff,stroke:none
    style F fill:#2e6da4,color:#fff,stroke:none
    style K fill:#2a7a4a,color:#fff,stroke:none
```

> **Ingestion** happens once when the PDF is uploaded — pages are loaded, split into overlapping chunks, embedded into vectors, and indexed in FAISS.
> **Retrieval** happens on every question — the query is embedded, the top-k most similar chunks are fetched, stuffed into the prompt alongside the question, and sent to the LLM for a grounded, cited answer.

---

## 🚀 Key Features

*   **PDF Document Parsing:** Upload any PDF and automatically load, parse, and structure its content.
*   **Intelligent Text Chunking:** Dynamically splits large documents using a character-based splitter to maintain context window sizing.
*   **Semantic Search & Retrieval:** Employs `sentence-transformers/all-MiniLM-L6-v2` embeddings combined with a **FAISS** vector database for super-fast similarity matching.
*   **Advanced QA Chain:** Builds a retrieval-augmented question-answering chain leveraging the state-of-the-art **Qwen-2.5-72B-Instruct** model.
*   **Source Citation:** Every answer lists the specific pages and text snippets retrieved as sources.
*   **Persistent Session Chat History:** Remembers conversational flow within the active session.

---

## 🛠️ Tech Stack & Libraries

*   **Frontend UI:** [Streamlit](https://streamlit.io/)
*   **LLM Orchestration:** [LangChain](https://github.com/langchain-ai/langchain)
*   **Embeddings:** [Hugging Face Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
*   **Vector Search Engine:** [FAISS CPU](https://github.com/facebookresearch/faiss)
