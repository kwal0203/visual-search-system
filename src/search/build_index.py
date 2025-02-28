import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PILImage
from torchvision import transforms

from src.embeddings.models import get_embedding_model
from src.data.mnist_loader import setup_mnist_database
from src.data.models import Image


def generate_embeddings(model, db, device="cuda"):
    """Generate embeddings for all images in the database."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Get all images
    images = db.query(Image).filter(Image.is_mnist == True).all()
    embeddings = []
    image_ids = []

    model.eval()
    with torch.no_grad():
        for img in tqdm(images, desc="Generating embeddings"):
            # Load and transform image
            image = PILImage.open(img.file_path)
            image = transform(image).unsqueeze(0).to(device)

            # Generate embedding
            embedding = model(image)
            embeddings.append(embedding.cpu().numpy())
            image_ids.append(img.image_id)

    embeddings = np.vstack(embeddings)
    return embeddings, image_ids


def build_search_index(embeddings, save_dir="models"):
    """Build and save FAISS index for fast similarity search."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add vectors to the index
    index.add(embeddings.astype(np.float32))

    # Save the index
    faiss.write_index(index, str(save_dir / "mnist_index.faiss"))
    print(f"Search index saved to {save_dir / 'mnist_index.faiss'}")
    return index


def load_search_system(
    model_path="models/embedding_model.pth",
    index_path="models/mnist_index.faiss",
    device="cuda",
):
    """Load the trained model and search index."""
    # Load model
    model = get_embedding_model(model_type="cnn")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Load index
    index = faiss.read_index(index_path)

    return model, index


if __name__ == "__main__":
    # Set up
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    db, _ = setup_mnist_database()

    # Load model
    model = get_embedding_model(model_type="cnn")
    model.load_state_dict(torch.load("models/embedding_model.pth"))
    model = model.to(device)

    # Generate embeddings and build index
    print("Generating embeddings for all images...")
    embeddings, image_ids = generate_embeddings(model, db, device)

    print("Building search index...")
    index = build_search_index(embeddings)

    # Save image IDs for later reference
    np.save("models/image_ids.npy", np.array(image_ids))
    print("Image IDs saved to models/image_ids.npy")
