"""
rag_pipeline.py

RAG (Retrieval-Augmented Generation) pipeline implementation.
Defines stages for loading, splitting, vector store creation, QA chain building, and querying.
"""

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain                # NEW: replaces RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os, tempfile

# Load environment variables
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# ── Stage 1: Load ──────────────────────────────────────────────
def loading(uploaded_file) -> list:
    """
    Saves the uploaded file temporarily, loads it using PyPDFLoader,
    unlinks the temp file, and returns the loaded pages.
    
    Args:
        uploaded_file: The uploaded PDF file.
        
    Returns:
        List of loaded PDF pages.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    os.unlink(temp_path)

    return pages

# ── Stage 2: Splitting into chunks ──────────────────────────────────────────────
def splitting(pages: list) -> list:
    """
    Splits the list of loaded pages into text chunks.
    
    Args:
        pages: List of loaded PDF pages.
        
    Returns:
        List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(pages)


# ── Stage 3: Feeding into Vector Stores ──────────────────────────────────────────────
def feed_vectr_store(chunks: list) -> FAISS:
    """
    Generates embeddings for the chunks and feeds them into a FAISS vector store.
    
    Args:
        chunks: List of document chunks.
        
    Returns:
        FAISS vector store object.
    """
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(embedding=embedding, documents=chunks)

# ── Stage 4: Chain building ──────────────────────────────────────────────

SYSTEM_PROMPT = """ You're a helpful assistant. Just answer from the below context and if you don't find the answer, just say that you couldn't find it

{context}"""

def build_qa_chain(vectorstore: FAISS):
    """
    Builds the question answering retrieval chain.
    
    Args:
        vectorstore: The FAISS vector store.
        
    Returns:
        The QA retrieval chain.
    """
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token
    )

    model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human', '{input}'),
    ])

    combine_docs_chain = create_stuff_documents_chain(model, prompt)

    return create_retrieval_chain(
        vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        combine_docs_chain
    )

# ── Stage 5: Query ──────────────────────────────────────────────
def query(chain, question: str):
    """
    Queries the QA retrieval chain with a question and formats the response.
    
    Args:
        chain: The QA retrieval chain.
        question: The user query string.
        
    Returns:
        A dictionary containing the answer and the extracted sources.
    """
    res = chain.invoke({'input': question})

    sources = []
    for doc in res['context']:
        page = doc.metadata.get('page', 0)
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = 0
        snippet = doc.page_content[:120].replace('\n', ' ')
        sources.append({'page': page+1, 'snippet': snippet})

    return ({'answer': res['answer'], 'sources': sources})