from dotenv import load_dotenv
load_dotenv()

from ingest import ingest_documents
from qa import ask_question

if __name__ == "__main__":
    ingest_documents("data")

    while True:
        q = input("\nAsk a question (or exit): ")
        if q.lower() == "exit":
            break

        answer, sources = ask_question(q)

        print("\nAnswer:", answer)
        if sources:
            print("\nSources:")
            for s in sources:
                print("-", s)
