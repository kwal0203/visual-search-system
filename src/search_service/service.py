from src.search_service.search import search_similar_images
from src.util.utils import load_config
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def search_index(query_image: str) -> list:
    config = load_config(config_path=CONFIG_PATH)
    similar_images = search_similar_images(query_image=query_image, config=config)
    return similar_images
