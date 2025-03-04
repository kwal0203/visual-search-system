from src.storage_service.mnist_loader import (
    setup_mnist_database,
    load_mnist,
)
from pathlib import Path

import os


def setup_storage(db_path: str, save_dir: str, raw_dir: str):
    # Raw data is stored in src/storage_service/raw
    # Processed data is stored in src/storage_service/processed
    # Trained model is stored in src/embedding_service/model

    # db_path = f"src/storage_service/mnist.db"
    # save_dir = "src/storage_service/processed"
    # raw_dir = "src/storage_service/raw"

    if not Path(db_path).exists():
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        setup_mnist_database(db_path=db_path)
        load_mnist(db_path=db_path, save_dir=save_dir, raw_dir=raw_dir)
