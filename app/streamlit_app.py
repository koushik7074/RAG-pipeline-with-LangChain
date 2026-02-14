import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_documents
from qa import ask_question

load_dotenv()

st.set_page_config(page_title="Doc Intelligence RAG", layout="wide")

st.title("LLM Document Intelligence System")
st.caption("Semantic search over documents using RAG")

# Ensure DB exists
ingest_documents("data")

# Input box
query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        answer, sources = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources")
        for s in sources:
            st.write(f"- {s}")
