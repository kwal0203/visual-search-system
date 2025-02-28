import torch
from pathlib import Path
from tqdm import tqdm

from src.embeddings.models import get_embedding_model, ContrastiveLoss


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
