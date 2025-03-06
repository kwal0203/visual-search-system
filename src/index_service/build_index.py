from src.embedding_service.models import get_embedding_model
from src.storage_service.service import get_images
from src.util.utils import load_training_config
from torchvision import transforms
from PIL import Image as PILImage
from tqdm import tqdm

import numpy as np
import torch
import faiss


def generate_embeddings(model_path: str, config_path: str):
    """Generate embeddings for all images in the database."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Get all images from training set
    images = get_images(dataset_split="train")
    embeddings = []
    image_ids = []

    config = load_training_config(config_path)
    model = get_embedding_model(
        model_type=config["model_type"],
        embedding_dim=config["embedding_dim"],
    )
    model.load_state_dict(torch.load(model_path))
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


def build_search_index(embeddings, index_path: str):
    """Build and save FAISS index for fast similarity search."""
    print("Building search index...")

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add vectors to the index
    index.add(embeddings.astype(np.float32))

    # Save the index
    index_path = str(index_path) + "/index.faiss"
    faiss.write_index(index, index_path)
    print(f"Search index saved to {index_path}")
