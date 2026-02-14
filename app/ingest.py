import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def ingest_documents(data_dir: str = "data", persist_dir: str = "chroma_db"):
    all_docs = []

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            path = os.path.join(data_dir, file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Indexed {len(chunks)} chunks from {data_dir}")
