from src.embedding_service.service import generate_embeddings
from src.index_service.build_index import build_search_index
from src.util.utils import load_config
from pathlib import Path

import faiss
import os

CONFIG_PATH = Path(__file__).parent / "config.json"


def build_index() -> None:
    # Generate embeddings and build index if they don't exist
    config = load_config(config_path=CONFIG_PATH)
    _index_path = Path(config["index_path"])
    if not _index_path.exists():
        print("Generating embeddings and building search index...")
        os.makedirs(config["index_path"], exist_ok=True)
        embeddings, image_ids = generate_embeddings()
        build_search_index(embeddings, config)
    else:
        print(f"Index already exists at {config['index_path']}...")


def load_index() -> faiss.IndexFlatL2:
    config = load_config(config_path=CONFIG_PATH)
    return faiss.read_index(str(config["index_path"]) + "/" + config["index_file_name"])
