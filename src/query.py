from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "faiss_index"

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve_context(db, question, k=3):
    docs = db.similarity_search(question, k=k)
    return docs

if __name__ == "__main__":
    db = load_vector_store()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        docs = retrieve_context(db, question)

        print("\nRetrieved Context:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Chunk {i} ---")
            print(doc.page_content)
