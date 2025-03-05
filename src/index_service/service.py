from src.index_service.build_index import generate_embeddings, build_search_index
from pathlib import Path

import faiss
import os


def build_index(model_path: str, config_path: str, index_path: str) -> None:
    # Generate embeddings and build index if they don't exist
    _index_path = Path(index_path)
    if not _index_path.exists():
        print("Generating embeddings and building search index...")
        os.makedirs(index_path, exist_ok=True)
        embeddings, image_ids = generate_embeddings(model_path, config_path)
        build_search_index(embeddings, index_path)
    else:
        print(f"Index already exists at {index_path}...")


def read_index(index_path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(index_path)
