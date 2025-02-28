from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize


class IndexBase(ABC):
    """Abstract base class for vector indexing and search."""

    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Build the index from embeddings."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        pass

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Add new embeddings to the index."""
        pass


class FaissIndex(IndexBase):
    """FAISS-based vector index."""

    def __init__(
        self,
        dim: int,
        index_type: str = "IVFFlat",
        metric: str = "cosine",
        nlist: int = 100,
        gpu: bool = False,
    ):
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.gpu = gpu
        self.index = None

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Build the index from embeddings."""
        if self.metric == "cosine":
            embeddings = normalize(embeddings)

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.nlist, faiss.METRIC_L2
            )
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        if self.gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )

        self.index.add_with_ids(embeddings, ids)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        if self.metric == "cosine":
            query = normalize(query.reshape(1, -1))

        distances, indices = self.index.search(query, k)
        return distances, indices

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Add new embeddings to the index."""
        if self.metric == "cosine":
            embeddings = normalize(embeddings)

        self.index.add_with_ids(embeddings, ids)


class ExactKNNIndex(IndexBase):
    """Exact k-NN search using brute force."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.embeddings = None
        self.ids = None

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Build the index from embeddings."""
        if self.metric == "cosine":
            self.embeddings = normalize(embeddings)
        else:
            self.embeddings = embeddings

        self.ids = ids

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        if self.metric == "cosine":
            query = normalize(query.reshape(1, -1))
            similarities = query @ self.embeddings.T
            distances = 1 - similarities
        else:
            distances = np.linalg.norm(self.embeddings - query.reshape(1, -1), axis=1)

        top_k_indices = np.argsort(distances.ravel())[:k]
        top_k_distances = distances.ravel()[top_k_indices]
        top_k_ids = self.ids[top_k_indices]

        return top_k_distances, top_k_ids

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Add new embeddings to the index."""
        if self.metric == "cosine":
            embeddings = normalize(embeddings)

        self.embeddings = np.vstack([self.embeddings, embeddings])
        self.ids = np.concatenate([self.ids, ids])


def get_index(index_type: str = "faiss", **kwargs) -> IndexBase:
    """Factory function to create vector indices."""
    indices = {
        "faiss": FaissIndex,
        "exact": ExactKNNIndex,
    }

    if index_type not in indices:
        raise ValueError(f"Unknown index type: {index_type}")

    return indices[index_type](**kwargs)
