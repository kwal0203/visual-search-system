# Nearest Neighbour Service that uses an image embedding passed in from the embedding service and then finds the nearest neighbour in the database. The service is generic enough to be used for exact nearest neighbour search, approximate nearest neighbour search,tree-based approximate nearest neighbour search clustering based approximate nearest neighbour search.

from src.embedding_service.service import embed_image

import torch

# class that represents a nearest neighbour service that accepts an image embedding and uses it to perform a nearest neighbour search on the system index table.


class NearestNeighbourService:
    def __init__(self, embedding_model: torch.nn.Module):
        self.embedding_model = embedding_model

    def search(self, embedding: torch.Tensor) -> list[str]:
        # Perform a nearest neighbour search on the system index table
        pass
