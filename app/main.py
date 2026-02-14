from ingest import ingest_document
from rag_pipeline import get_rag_chain
from dotenv import load_dotenv

load_dotenv()

def ask_question(query: str):
    retriever, llm = get_rag_chain()

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using only the context below.
    If unsure, say you don't know.

    Context: {context}
    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    # Run once to index
    ingest_document("data/sample.pdf")

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", ask_question(q))
