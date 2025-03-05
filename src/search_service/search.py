from src.embedding_service.models import get_embedding_model
from src.index_service.service import read_index
from src.storage_service.models import Image
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from torchvision import transforms
from PIL import Image as PILImage
from typing import Optional
from pathlib import Path

import torch
import json
import os


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


def search_similar_images(
    query_image: Image,
    db_path: str,
    model_path: str,
    index_path: str,
    config_path: str,
    k: int = 5,
):
    """Find k most similar images to the query image."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # Generate query embedding
    with torch.no_grad():
        config = load_training_config(config_path)
        image = PILImage.open(query_image.file_path)

        image_path = Path("/home/kwal0203/results") / f"query.png"
        image.save(image_path)

        image = transform(image).unsqueeze(0).to(config["device"])
        model = get_embedding_model(
            model_type=config["model_type"],
            embedding_dim=config["embedding_dim"],
        )
        model.load_state_dict(torch.load(model_path))
        model = model.to(config["device"])
        model.eval()
        query_embedding = model(image).cpu().numpy()

    # Search in the index
    print(f"READING {index_path + "/index.faiss"}")
    index = read_index(index_path + "/index.faiss")
    distances, indices = index.search(query_embedding, k)
    print(f"Similar indices: {indices}")
    print(f"Distances: {distances}")

    # Get the corresponding images
    similar_images = []
    for idx in indices[0]:
        img = db.query(Image).filter(Image.image_id == int(idx + 1)).first()
        if img:
            similar_images.append(
                {
                    "image_id": img.image_id,
                    "file_path": img.file_path,
                    "digit": img.digit_label,
                }
            )

        # save image as .png in /tmp/results using PIL
        image_path = Path("/home/kwal0203/results") / f"{img.image_id}.png"
        os.makedirs(image_path.parent, exist_ok=True)
        print(f"{img.file_path}")
        source_image = PILImage.open(img.file_path)
        source_image.save(image_path)
    db.close()

    return similar_images
