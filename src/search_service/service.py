from src.search_service.search import search_similar_images
from src.util.utils import load_config


def search_index(
    query_image: str,
    db_path: str,
    model_path: str,
    index_path: str,
    config_path: str,
    k: int = 5,
) -> list:
    config = load_config(config_path)
    similar_images = search_similar_images(
        query_image=query_image,
        db_path=db_path,
        model_path=model_path,
        index_path=index_path,
        config_path=config_path,
        k=k,
    )
    return similar_images
