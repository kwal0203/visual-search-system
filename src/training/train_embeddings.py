import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.embeddings.models import get_embedding_model, ContrastiveLoss
from src.data.mnist_loader import setup_mnist_database, generate_contrastive_pairs


class ContrastivePairDataset(Dataset):
    """Dataset for training with contrastive pairs."""

    def __init__(self, pairs, labels, db):
        self.pairs = pairs
        self.labels = labels
        self.db = db
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]

        # Get images from database
        img1 = self.db.query(Image).filter(Image.image_id == pair[0]).first()
        img2 = self.db.query(Image).filter(Image.image_id == pair[1]).first()

        # Load and transform images
        img1 = Image.open(img1.file_path)
        img2 = Image.open(img2.file_path)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.FloatTensor([label])


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    db, _ = setup_mnist_database()

    # Generate pairs and create dataset/dataloader
    print("Generating contrastive pairs for training...")
    pairs, labels = generate_contrastive_pairs(db, num_pairs=50000)
    dataset = ContrastivePairDataset(pairs, labels, db)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = train_embedding_model(
        db=db,
        pairs=pairs,
        labels=labels,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
    )
