from src.data.random_sampler import BalancedRandomPairBatchSampler
from src.search.search import search_similar_images
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path

from src.data.mnist_loader import (
    setup_mnist_database,
    load_mnist,
    generate_contrastive_pairs,
)
from src.data.contrastive_dataset import ContrastivePairDatasetMNIST
from src.embeddings.train import train_embedding_model
from src.search.build_index import (
    generate_embeddings,
    build_search_index,
)

import matplotlib.pyplot as plt
import torch
import os


# Get the project root directory (where src directory is located)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def display_search_results(query_image_path, similar_images, num_results=5):
    """Display query image and its similar images."""
    plt.figure(figsize=(12, 4))

    # Show query image
    plt.subplot(1, num_results + 1, 1)
    query_img = PILImage.open(query_image_path)
    plt.imshow(query_img, cmap="gray")
    plt.title("Query Image")
    plt.axis("off")

    # Show similar images
    for i, img_info in enumerate(similar_images[:num_results], 1):
        plt.subplot(1, num_results + 1, i + 1)
        similar_img = PILImage.open(img_info["file_path"])
        plt.imshow(similar_img, cmap="gray")
        plt.title(f"Similar {i}\nDigit: {img_info['digit']}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_dataloader(mnist_user, db_path):
    load_mnist(db_path, mnist_user)

    # Create dataset and DataLoader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = ContrastivePairDatasetMNIST(
        db_path=db_path,
        dataset_split="train",
        transform=transform,
    )
    test_dataset = ContrastivePairDatasetMNIST(
        db_path=db_path,
        dataset_split="test",
        transform=transform,
    )

    train_sampler = BalancedRandomPairBatchSampler(train_dataset)
    test_sampler = BalancedRandomPairBatchSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    return train_dataloader, test_dataloader


def main():
    # Set up
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    db_path = "training_outputs/data"
    os.makedirs(db_path, exist_ok=True)
    setup_mnist_database(db_path)
    train_dataloader, test_dataloader = get_dataloader(db_path)

    # Check if model exists, if not train it
    model_path = PROJECT_ROOT / "training_outputs" / "model" / "embedding_model.pth"
    model_path.parent.mkdir(
        exist_ok=True
    )  # Create models directory if it doesn't exist

    if not model_path.exists():
        print("Training embedding model...")
        config_path = PROJECT_ROOT / "src" / "embeddings" / "config.json"
        print(f"Using config from: {config_path}")
        model = train_embedding_model(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            config_path=str(config_path),
        )
    else:
        print("Loading existing model...")
        model = torch.load(model_path)
        model = model.to(device)

    # # Generate embeddings and build index if they don't exist
    # index_path = Path("models/mnist_index.faiss")
    # if not index_path.exists():
    #     print("Generating embeddings and building search index...")
    #     embeddings, image_ids = generate_embeddings(model, db, device)
    #     index = build_search_index(embeddings)
    # else:
    #     print("Loading existing index...")
    #     index = torch.load(index_path)

    # # Perform a sample search
    # print("\nPerforming sample search...")
    # # Get a random test image
    # query_image = (
    #     db.query(DBImage)
    #     .filter(DBImage.is_mnist == True, DBImage.dataset_split == "test")
    #     .first()
    # )

    # similar_images = search_similar_images(
    #     query_image.image_id, db, model, index, k=5, device=device
    # )

    # # Display results
    # display_search_results(query_image.file_path, similar_images)


if __name__ == "__main__":
    main()
