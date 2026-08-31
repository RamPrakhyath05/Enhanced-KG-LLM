import os
import pickle
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.bm25 = None
        self.documents = []

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    def build(self, documents):
        self.documents = documents

        tokenized_documents = [
            self.tokenize(doc["text"])
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=10):
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[index]

            results.append({
                "id": document["id"],
                "name": document["name"],
                "score": float(scores[index]),
            })

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]
