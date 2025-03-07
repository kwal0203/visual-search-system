from src.embedding_service.models import get_embedding_model
from src.index_service.service import load_index
from src.storage_service.models import Image
from src.util.utils import get_transform
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from PIL import Image as PILImage
from pathlib import Path
from typing import Dict
import glob

import torch
import os


def _get_next_result_dir(base_path: str) -> str:
    """Get the next available numbered result directory, ensuring sequential numbering from 1."""
    os.makedirs(base_path, exist_ok=True)
    existing_dirs = [
        d
        for d in glob.glob(os.path.join(base_path, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ]

    if not existing_dirs:
        return os.path.join(base_path, "1")

    # Sort directories numerically
    dir_numbers = sorted([int(os.path.basename(d)) for d in existing_dirs])
    next_number = dir_numbers[-1] + 1
    return os.path.join(base_path, str(next_number))


def search_similar_images(query_image: Image, config: Dict):
    """Find k most similar images to the query image."""
    transform = get_transform(name=config["transform"])

    # Create a new numbered results directory
    base_results_dir = Path("/home/kwal0203/results")
    result_dir = _get_next_result_dir(str(base_results_dir))
    os.makedirs(result_dir, exist_ok=True)

    # Generate query embedding
    with torch.no_grad():
        image = PILImage.open(query_image.file_path)

        # Save query image in the numbered directory
        image_path = Path(result_dir) / "query.png"
        image.save(image_path)

        image = transform(image).unsqueeze(0).to(config["device"])
        model = get_embedding_model(
            model_type=config["model_type"],
            embedding_dim=config["embedding_dim"],
        )
        model.load_state_dict(torch.load(config["model_path"]))
        model = model.to(config["device"])
        model.eval()
        query_embedding = model(image).cpu().numpy()

    # Search in the index
    index = load_index()
    distances, indices = index.search(query_embedding, config["k"])

    # Get the corresponding images. Embedding index numbers off by 1 compared with database image ids.
    engine = create_engine(f"sqlite:///{config['db_path']}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

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

            # Save image in the numbered directory
            image_path = os.path.join(result_dir, f"result_{img.image_id}.png")
            source_image = PILImage.open(img.file_path)
            source_image.save(image_path)
    db.close()

    return similar_images
