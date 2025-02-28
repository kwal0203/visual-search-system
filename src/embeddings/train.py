import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
import json
from tqdm import tqdm

from src.embeddings.models import get_embedding_model


class ContrastiveLoss(nn.Module):
    """Contrastive loss with support for different similarity metrics."""

    def __init__(
        self,
        temperature: float = 0.07,
        similarity: str = "cosine",
        reduction: str = "mean",
        debug: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        self.reduction = reduction
        self.debug = debug

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
        # Validate input shapes
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embedding shapes must match. Got {embedding1.shape} and {embedding2.shape}"
            )

        if labels is not None:
            # Squeeze labels to make it 1D if it's 2D
            labels = labels.squeeze()
            if labels.shape[0] != embedding1.shape[0]:
                raise ValueError(
                    f"Number of labels ({labels.shape[0]}) must match batch size ({embedding1.shape[0]})"
                )

        # Debug prints
        if self.debug:
            print(f"Shape of embedding1: {embedding1.shape}")
            print(f"Shape of embedding2: {embedding2.shape}")
            if labels is not None:
                print(f"Shape of labels: {labels.shape}")

        # Concatenate embeddings for combined processing
        embeddings = torch.cat([embedding1, embedding2], dim=0)
        batch_size = embedding1.shape[0]

        if self.debug:
            print(f"Shape of concatenated embeddings: {embeddings.shape}")

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        if self.similarity == "cosine":
            similarity_matrix = torch.matmul(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        if self.debug:
            print(f"Shape of similarity matrix: {similarity_matrix.shape}")

        # Scale similarities
        similarity_matrix = similarity_matrix / self.temperature

        # Create labels if not provided (self-supervised case)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings.device)

        # Extend labels to match concatenated embeddings
        labels = torch.cat([labels, labels])
        if self.debug:
            print(f"Shape of extended labels: {labels.shape}")

        # Create positive mask
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        if self.debug:
            print(f"Shape of positive mask: {positive_mask.shape}")
        positive_mask.fill_diagonal_(
            False
        )  # Set diagonal to False to exclude self-pairs

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


def load_training_config(config_path: Optional[str] = None) -> dict:
    """Load training configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file. If None, uses default config.

    Returns:
        Dictionary containing training configuration.
    """
    default_config_path = Path(__file__).parent / "config.json"

    # Load default config first
    try:
        with open(default_config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Default config file not found at {default_config_path}"
        )

    # If custom config provided, update default config with custom values
    if config_path is not None:
        try:
            with open(config_path, "r") as f:
                custom_config = json.load(f)
                config.update(custom_config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Custom config file not found at {config_path}")

    return config


def train_embedding_model(dataloader, config_path: Optional[str] = None):
    """Train the embedding model using contrastive pairs.

    Args:
        dataloader: DataLoader instance
        config_path: Path to JSON config file containing training parameters.
                    If None, uses default config file.
                    The JSON file should contain the following keys:
                    - num_epochs: Number of training epochs
                    - embedding_dim: Dimension of the embedding space
                    - learning_rate: Learning rate for optimization
                    - device: Device to run the training on
                    - model_type: Type of embedding model to use
                    - contrastive_loss_temp: Temperature for contrastive loss
                    - contrastive_loss_similarity: Similarity metric for contrastive loss
                    - contrastive_loss_reduction: Reduction method for contrastive loss
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
        temperature=training_config["contrastive_loss_temp"],
        similarity=training_config["contrastive_loss_similarity"],
        reduction=training_config["contrastive_loss_reduction"],
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
