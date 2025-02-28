from abc import ABC, abstractmethod

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


class CNNEmbedding(EmbeddingModel):
    """Simple CNN for learning digit embeddings (MNIST-specific)."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__(embedding_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNEmbeddingFromChatGPT(nn.Module):
    """Improved CNN for contrastive learning with normalized embeddings."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after Conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm after Conv2

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.bn3 = nn.BatchNorm1d(256)  # BatchNorm for FC layer
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.leaky_relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x  # Normalized in loss function


def get_embedding_model(model_type: str = "resnet", **kwargs) -> EmbeddingModel:
    """Factory function to create embedding models."""
    models = {
        "resnet": ResNetEmbedding,
        "cnn": CNNEmbedding,
        "cnn_chatgpt": CNNEmbeddingFromChatGPT,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)
