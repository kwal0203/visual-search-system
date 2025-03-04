from src.embedding_service.random_sampler import BalancedRandomPairBatchSampler
from src.storage_service.service import setup_storage
from src.search.search import search_similar_images
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path
from src.storage_service.mnist_loader import (
    setup_mnist_database,
    load_mnist,
    generate_contrastive_pairs,
)
from src.embedding_service.service import train_model
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    db_path = f"src/storage_service/mnist.db"
    save_dir = "src/storage_service/processed"
    raw_dir = "src/storage_service/raw"
    setup_storage(db_path=db_path, save_dir=save_dir, raw_dir=raw_dir)

    config_path = PROJECT_ROOT / "src" / "embedding_service" / "config.json"
    model_path = (
        PROJECT_ROOT / "src" / "embedding_service" / "model" / "embedding_model.pth"
    )
    train_model(
        db_path=db_path,
        config_path=config_path,
        model_path=model_path,
    )

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
