import numpy as np
import faiss
from pandas import DataFrame


class ComputeSimilarity:
    """
    Semantic similarity search using FAISS
    """
    def __init__(self, embeddings: DataFrame):
        vectors = embeddings.values.astype("float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.ids = list(embeddings.index)

    def search(self, query_id: str, k: int = 10) -> list[dict]:
        idx = self.ids.index(query_id)
        query = self.index.reconstruct(idx).reshape(1, -1)
        distances, indices = self.index.search(query, k + 1)  # +1 to exclude self
        results = []
        for dist, i in zip(distances[0], indices[0]):
            if self.ids[i] != query_id:  # exclude self
                results.append({"id": self.ids[i], "score": float(dist)})
        return results[:k]