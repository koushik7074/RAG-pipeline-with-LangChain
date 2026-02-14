from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

def get_rag_components(persist_dir="chroma_db"):
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Generator LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return retriever, llm
