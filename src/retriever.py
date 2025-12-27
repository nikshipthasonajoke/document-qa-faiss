import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self.load()

    def load(self):
        if os.path.exists(self.index_path):
            self.db = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def search(self, question, k=3, threshold=1.3):
        """
        Returns empty list ONLY if question is truly out-of-scope.
        Threshold relaxed for small documents.
        """
        if not self.db:
            return []

        results = self.db.similarity_search_with_score(question, k=k)

        # Out-of-scope check
        if not results or results[0][1] > threshold:
            return []

        return [doc.page_content for doc, _ in results]
