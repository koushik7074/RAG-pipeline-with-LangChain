from ingest import ingest_documents
from rag_pipeline import get_rag_components
from dotenv import load_dotenv

load_dotenv()

def ask_question(query: str):
    retriever, llm = get_rag_components()

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful AI assistant.

    Use ONLY the provided context to answer the question.
    If the answer cannot be found, say: "I don't know".

    Be concise and factual.

    Context:
    {context}

    Question: {query}
    Answer:
    """


    response = llm.invoke(prompt)

    # Extract sources
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} "
        f"(page {doc.metadata.get('page', 'N/A')})"
        for doc in docs
    ]


    return response.content, list(set(sources))



if __name__ == "__main__":
    ingest_documents("data")

    while True:
        q = input("\nAsk a question (or exit): ")
        if q.lower() == "exit":
            break

        answer, sources = ask_question(q)

        print("\nAnswer:", answer)
        print("\nSources:")
        for s in sources:
            print("-", s)