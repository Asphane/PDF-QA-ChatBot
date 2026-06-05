"""
app.py

Streamlit application entry point. Handles the UI, sidebar file uploads,
chat messaging, and user query dispatching to the RAG pipeline.
"""

from rag_pipeline import query
import streamlit as st
from rag_pipeline import loading, splitting, feed_vectr_store, build_qa_chain
from chat_history import ChatHistoryMsg

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Question Answering")

# ── Initialize session state ──
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatHistoryMsg()

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# ── Sidebar: upload + process ──
with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded and st.button('Process PDF'):
        with st.spinner("Processing the PDF"):
            pages = loading(uploaded)
            chunks = splitting(pages)
            store = feed_vectr_store(chunks)
            
            st.session_state.qa_chain = build_qa_chain(store)
            st.session_state.chat_history = ChatHistoryMsg()

            st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages")

# ── Chat interface ─
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.role):
        st.write(msg.content)

        if msg.sources:
            with st.expander("📎 Sources"):
                for s in msg.sources:
                    st.caption(f"Page {s['page']}: …{s['snippet']}…")

if question:= st.chat_input("Ask Something about pdf"):
    if not st.session_state.qa_chain:
        st.warning("Please upload and process a PDF first.")
    
    else:
        st.session_state.chat_history.add('user', question)
        with st.chat_message('User'):
            st.write(question)

        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                response = query(st.session_state.qa_chain, question)

            st.write(response['answer'])
            if response['sources']:
                with st.expander("📎 Sources"):
                    for s in response['sources']:
                        st.caption(f"Page {s['page']}: ...{s['snippet']}...")

                st.session_state.chat_history.add(
                    'assistant', response['answer'],
                    response['sources']
                )