from src.embedding_service.models import get_embedding_model
from src.util.utils import load_config, get_transform
from src.index_service.service import load_index
from src.storage_service.models import Image
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from PIL import Image as PILImage
from pathlib import Path

import torch
import os

CONFIG_PATH = Path(__file__).parent / "config.json"


def search_similar_images(query_image: Image):
    """Find k most similar images to the query image."""
    config = load_config(config_path=CONFIG_PATH)
    transform = get_transform(name=config["transform"])

    # Generate query embedding
    with torch.no_grad():
        image = PILImage.open(query_image.file_path)

        image_path = Path("/home/kwal0203/results") / f"query.png"
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

    print(f"Similar indices: {indices}")
    print(f"Distances: {distances}")

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

        # save image as .png in /tmp/results using PIL
        image_path = config["save_dir"].format(image_id=img.image_id)
        os.makedirs(Path(image_path).parent, exist_ok=True)
        source_image = PILImage.open(img.file_path)
        source_image.save(image_path)
    db.close()

    return similar_images
