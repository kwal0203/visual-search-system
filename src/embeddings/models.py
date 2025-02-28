from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingModel(nn.Module, ABC):
    """Abstract base class for embedding models."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate embeddings."""
        pass

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings for inference."""
        with torch.no_grad():
            embedding = self.forward(x)
            return F.normalize(embedding, p=2, dim=1)


class ResNetEmbedding(EmbeddingModel):
    """ResNet-based embedding model."""

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__(embedding_dim)

        # Load backbone
        backbone_fn = getattr(models, backbone)
        self.backbone = backbone_fn(pretrained=pretrained)

        # Replace final layer with embedding projection
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ContrastiveLoss(nn.Module):
    """Contrastive loss with support for different similarity metrics."""

    def __init__(
        self,
        temperature: float = 0.07,
        similarity: str = "cosine",
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        self.reduction = reduction

    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Optional tensor of shape (batch_size,) for supervised contrastive loss
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        if self.similarity == "cosine":
            similarity_matrix = torch.matmul(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Scale similarities
        similarity_matrix = similarity_matrix / self.temperature

        # Create labels if not provided (self-supervised case)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings.device)

        # Create positive mask
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(False)

        # Create negative mask
        negative_mask = ~positive_mask

        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Compute mean of positive similarities
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(
            dim=1
        ).clamp(min=1)

        # Compute loss
        loss = -mean_log_prob_pos

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def get_embedding_model(model_type: str = "resnet", **kwargs) -> EmbeddingModel:
    """Factory function to create embedding models."""
    models = {
        "resnet": ResNetEmbedding,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)
