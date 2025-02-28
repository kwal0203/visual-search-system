import torch
from PIL import Image as PILImage
from torchvision import transforms
from src.data.models import Image


def search_similar_images(query_image_id, db, model, index, k=5, device="cuda"):
    """Find k most similar images to the query image."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Get query image
    query_img = db.query(Image).filter(Image.image_id == query_image_id).first()
    if not query_img:
        raise ValueError("Query image not found")

    # Generate query embedding
    image = PILImage.open(query_img.file_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(image).cpu().numpy()

    # Search in the index
    distances, indices = index.search(query_embedding, k)

    # Get the corresponding images
    similar_images = []
    for idx in indices[0]:
        img = db.query(Image).filter(Image.image_id == idx).first()
        if img:
            similar_images.append(
                {
                    "image_id": img.image_id,
                    "file_path": img.file_path,
                    "digit": img.digit_label,
                }
            )

    return similar_images
