from src.embedding_service.models import get_embedding_model
from src.embedding_service.util import plot_losses
from torch.utils.data import DataLoader
from typing import Dict
from pathlib import Path
from tqdm import tqdm

import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        margin=1.0,
        mode="pairs",
        device="cpu",
        similarity="euclidean",
        reduction="mean",
    ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.device = device
        self.similarity = similarity
        self.reduction = reduction

    def forward(self, embeddings, labels=None):
        """
        Compute contrastive loss for pairs.

        Args:
            embeddings (torch.Tensor): tensor of shape (B, D)
            labels (torch.Tensor, optional): Tensor with values 0 or 1 indicating similarity

        Returns:
            torch.Tensor: Contrastive loss scalar
        """

        # Form pairs if labels are not provided
        if labels is None:
            x1, x2, labels = self.form_pairs(embeddings)
            x1, x2 = x1.to(self.device), x2.to(self.device)
            labels.to(self.device)

        dist = self.distance(x1, x2)

        # Contrastive loss calculation
        positive_loss = labels * dist
        negative_loss = (1 - labels) * torch.clamp(self.margin - dist, min=0.0)
        loss_contrastive = torch.mean(positive_loss + negative_loss)

        return loss_contrastive

    def form_pairs(self, embeddings):
        """
        Form all possible pairs from x1 and x2, where corresponding batch
        positions are positive pairs and all others are negative.

        Returns:
            repeated_x1 (torch.Tensor): Expanded x1 of shape (B², D)
            repeated_x2 (torch.Tensor): Expanded x2 of shape (B², D)
            labels (torch.Tensor): Labels of shape (B²,)
        """
        batch_size, embedding_size = embeddings.shape

        # Labels: Identity matrix reshaped to indicate positive pairs
        labels = torch.eye(batch_size, device=self.device, dtype=torch.long).reshape(-1)

        # Expand tensors
        repeated_x1 = embeddings.repeat(batch_size, 1)
        repeated_x2 = embeddings.repeat_interleave(batch_size, dim=0)

        return repeated_x1, repeated_x2, labels

    def distance(self, x1, x2):
        """Compute squared L2 distance between vectors."""
        return torch.pow(x1 - x2, 2).sum(1)


def train_embedding_model(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: Dict,
):
    """Train the embedding model using contrastive pairs.

    Args:
        train_dataloader: DataLoader instance
        test_dataloader: DataLoader instance
        config_path: Path to JSON config file containing training parameters.
    """
    model = get_embedding_model(
        model_type=config["model_type"],
        embedding_dim=config["embedding_dim"],
    )
    model = model.to(config["device"])
    model.train()

    criterion = ContrastiveLoss(
        margin=config["margin"],
        mode=config["mode"],
        device=config["device"],
        similarity=config["contrastive_loss_similarity"],
        reduction=config["contrastive_loss_reduction"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print(f"Starting training for {config['num_epochs']} epochs...")
    epoch_losses = []
    for epoch in range(config["num_epochs"]):
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"
        )

        epoch_loss = 0
        for x, _ in progress_bar:
            x = x.to(config["device"])

            # Forward pass
            embeddings = model(x)
            loss = criterion(embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Average Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    # Save the trained model
    save_dir = Path("src/embedding_service/model")
    torch.save(model.state_dict(), save_dir / "embedding_model.pth")
    print(f"Model saved to {save_dir / 'embedding_model.pth'}")

    # Result logging
    save_dir = Path("src/embedding_service/results")
    save_dir.mkdir(exist_ok=True)
    plot_losses(
        epoch_losses=epoch_losses,
        save_path=save_dir / "loss.png",
    )
