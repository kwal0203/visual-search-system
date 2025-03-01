import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
import json
from tqdm import tqdm
import math

from src.embeddings.models import get_embedding_model


class ContrastiveLoss(nn.Module):
    """Contrastive loss with support for different similarity metrics."""

    def __init__(
        self,
        base_temperature: float = 0.07,
        similarity: str = "cosine",
        reduction: str = "mean",
        margin: float = 0.5,  # margin for negative pairs
    ):
        super().__init__()
        self.base_temperature = base_temperature
        self.similarity = similarity
        self.reduction = reduction
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between pairs of embeddings.

        Args:
            embedding1: First set of embeddings of shape (batch_size, embedding_dim)
            embedding2: Second set of embeddings of shape (batch_size, embedding_dim)
            labels: Optional tensor of shape (batch_size,) for supervised contrastive loss
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embedding shapes must match. Got {embedding1.shape} and {embedding2.shape}"
            )

        batch_size = embedding1.shape[0]
        # Adjust temperature based on batch size
        self.temperature = self.base_temperature * (1 + math.log(batch_size) / 10)

        # Concatenate and normalize embeddings
        embeddings = torch.cat([embedding1, embedding2], dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        if self.similarity == "cosine":
            similarity_matrix = torch.matmul(embeddings, embeddings.T)
            similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        elif self.similarity == "euclidean":
            # Compute pairwise L2 distances
            # Using ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            square_norm = torch.sum(embeddings**2, dim=1, keepdim=True)
            distances = (
                square_norm + square_norm.T - 2 * torch.matmul(embeddings, embeddings.T)
            )
            distances = torch.clamp(
                distances, min=0.0
            )  # Ensure non-negative due to numerical precision
            # Convert distances to similarities (negative distance, scaled to similar range as cosine)
            similarity_matrix = -torch.sqrt(
                distances + 1e-8
            )  # Add epsilon to avoid sqrt(0)
            # Normalize to roughly [-1, 1] range like cosine similarity
            similarity_matrix = similarity_matrix / math.sqrt(embeddings.shape[1])
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Scale similarities with improved numerical stability
        # similarity_matrix = similarity_matrix / self.temperature
        max_val = torch.max(similarity_matrix)
        similarity_matrix = (similarity_matrix - max_val) * 0.9

        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
        epsilon = torch.max(exp_sim_sum) * 1e-6
        log_prob = similarity_matrix - torch.log(exp_sim_sum + epsilon)

        # Create labels if not provided (self-supervised case)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings.device)
        labels = torch.cat([labels, labels])

        # Create positive and negative masks
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(False)
        negative_mask = ~positive_mask

        # Compute positive pair loss (attraction)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(
            dim=1
        ).clamp(min=1)

        # Compute negative pair loss with margin (repulsion)
        # Only apply margin to pairs that are closer than margin
        neg_similarities = similarity_matrix * negative_mask
        margin_violation = F.relu(
            neg_similarities + self.margin
        )  # Only penalize pairs closer than -margin
        neg_loss = (margin_violation * negative_mask).sum(dim=1) / negative_mask.sum(
            dim=1
        ).clamp(min=1)

        # Combine both losses
        loss = -mean_log_prob_pos + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def load_training_config(config_path: Optional[str] = None) -> dict:
    """Load training configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file. If None, uses default config.

    Returns:
        Dictionary containing training configuration.
    """
    config_path = (
        Path(config_path) if config_path else Path(__file__).parent / "config.json"
    )

    try:
        with config_path.open("r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    return config


def train_embedding_model(dataloader, config_path: Optional[str] = None):
    """Train the embedding model using contrastive pairs.

    Args:
        dataloader: DataLoader instance
        config_path: Path to JSON config file containing training parameters.
    """
    # Load configuration from JSON file
    training_config = load_training_config(config_path)

    # Initialize model and move to device
    model = get_embedding_model(
        model_type=training_config["model_type"],
        embedding_dim=training_config["embedding_dim"],
    )
    model = model.to(training_config["device"])

    criterion = ContrastiveLoss(
        base_temperature=training_config["contrastive_loss_temp"],
        similarity=training_config["contrastive_loss_similarity"],
        reduction=training_config["contrastive_loss_reduction"],
        margin=training_config["contrastive_loss_margin"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config["learning_rate"]
    )

    print(f"Starting training for {training_config['num_epochs']} epochs...")
    for epoch in range(training_config["num_epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']}"
        )

        for img1, img2, labels in progress_bar:
            print(f"  -- SHAPE img1: {img1.shape}")
            print(f"  -- SHAPE img2: {img2.shape}")
            print(f"  -- SHAPE labels: {labels.shape}")

            # Move data to device
            img1 = img1.to(training_config["device"])
            img2 = img2.to(training_config["device"])
            labels = labels.to(training_config["device"])

            # Forward pass
            embedding1 = model(img1)
            embedding2 = model(img2)
            loss = criterion(embedding1, embedding2, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch+1}/{training_config['num_epochs']}, Average Loss: {avg_loss:.4f}"
        )

    # Save the trained model
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "embedding_model.pth")
    print(f"Model saved to {save_dir / 'embedding_model.pth'}")

    return model
