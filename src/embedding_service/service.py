from src.embedding_service.models import get_embedding_model
from PIL import Image
import torch


def embed_image(image_path: str, model_path: str) -> torch.Tensor:
    embedding_model = torch.load(model_path)
    image = Image.open(image_path)
    # Send the new image to object store
    return embedding_model.embed_image(image)
