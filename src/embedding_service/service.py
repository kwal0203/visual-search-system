from src.embedding_service.train import _train_embedding_model
from src.embedding_service.models import get_embedding_model
from src.storage_service.service import get_images
from src.util.utils import load_config, get_transform
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

CONFIG_PATH = Path(__file__).parent / "config.json"


def generate_embeddings():
    """API ENDPOINT: Generate embeddings for all images in the database."""
    config = load_config(config_path=CONFIG_PATH)
    transform = get_transform(name=config["transform"])

    # Get all images from training set
    images = get_images(dataset_split="train")
    embeddings = []
    image_ids = []

    model = get_embedding_model(
        model_type=config["model_type"],
        embedding_dim=config["embedding_dim"],
    )
    model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(config["device"])
    model.eval()
    with torch.no_grad():
        for img in tqdm(images, desc="Generating embeddings"):
            # Load and transform image
            image = PILImage.open(img.file_path)
            image = transform(image).unsqueeze(0).to(config["device"])

            # Generate embedding
            embedding = model(image)
            embeddings.append(embedding.cpu().numpy())
            image_ids.append(img.image_id)

    embeddings = np.vstack(embeddings)
    return embeddings, image_ids


def train_embedding_model():
    """API ENDPOINT"""
    config = load_config(config_path=CONFIG_PATH)
    if Path(config["model_path"]).exists():
        print("Model exists. Load from model_path...")
        return

    print("Model does not exist. Training new embedding model...")
    _train_embedding_model(config=config)
