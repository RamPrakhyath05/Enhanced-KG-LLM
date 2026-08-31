import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.embeddings = None
        self.documents = []

    def build(self, documents, batch_size=64):
        self.documents = documents

        texts = [
            document["text"]
            for document in documents
        ]

        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query, top_k=10):

        if self.embeddings is None:
            raise RuntimeError(
                "Dense index has not been built."
            )

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1][:top_k]

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
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "documents": self.documents,
                },
                f,
            )

    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.embeddings = data["embeddings"]
        self.documents = data["documents"]
