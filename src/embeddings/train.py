import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from src.embeddings.models import get_embedding_model


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
        # Concatenate embeddings for combined processing
        embeddings = torch.cat([embedding1, embedding2], dim=0)
        batch_size = embedding1.shape[0]

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

        # Extend labels to match concatenated embeddings
        labels = torch.cat([labels, labels])

        # Create positive mask
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(False)

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


def train_embedding_model(
    dataloader,
    num_epochs=10,
    embedding_dim=64,
    learning_rate=0.001,
    device="cuda",
):
    """Train the embedding model using contrastive pairs.

    Args:
        dataloader: DataLoader instance
        num_epochs: Number of training epochs
        embedding_dim: Dimension of the embedding space
        learning_rate: Learning rate for optimization
        device: Device to run the training on
    """
    # Initialize model and move to device
    model = get_embedding_model(model_type="cnn", embedding_dim=embedding_dim)
    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for img1, img2, labels in progress_bar:
            # Move data to device
            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "embedding_model.pth")
    print(f"Model saved to {save_dir / 'embedding_model.pth'}")

    return model
