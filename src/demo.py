from src.storage_service.service import setup_storage
from src.index_service.service import build_index
from PIL import Image as PILImage
from pathlib import Path
from src.embedding_service.service import train_model
from src.search_service.search import search_similar_images

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

    # Setup object storage
    db_path = f"src/storage_service/mnist.db"
    save_dir = "src/storage_service/processed"
    raw_dir = "src/storage_service/raw"
    setup_storage(db_path=db_path, save_dir=save_dir, raw_dir=raw_dir)

    # Train embedding model if required
    config_path = PROJECT_ROOT / "src" / "embedding_service" / "config.json"
    model_path = (
        PROJECT_ROOT / "src" / "embedding_service" / "model" / "embedding_model.pth"
    )
    train_model(
        db_path=db_path,
        config_path=config_path,
        model_path=model_path,
    )

    # Generate embeddings and build index if they don't exist
    index_path = "src/index_service/models/mnist_index"
    build_index(model_path=model_path, config_path=config_path, index_path=index_path)

    # Perform a sample search
    print("\nPerforming sample search...")
    from sqlalchemy import create_engine, func
    from sqlalchemy.orm import sessionmaker
    from src.storage_service.models import Image as ImageModel

    # Get a random test image
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    query_image = (
        db.query(ImageModel)
        .filter(ImageModel.is_mnist == True, ImageModel.dataset_split == "test")
        .order_by(func.random())
        .first()
    )
    print(f"Query image: {query_image.digit_label}")
    db.close()

    similar_images = search_similar_images(
        query_image=query_image,
        db_path=db_path,
        model_path=model_path,
        index_path=index_path,
        config_path=config_path,
        k=5,
    )


if __name__ == "__main__":
    main()
