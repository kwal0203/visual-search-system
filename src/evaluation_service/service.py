from src.evaluation_service.metrics import RankingMetrics
from src.storage_service.service import get_test_dataset
from src.search_service.search import search_index
from src.util.utils import load_config
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def evaluate(method: str = None):
    config = load_config(CONFIG_PATH)
    test_dataset = get_test_dataset(dataset_name="mnist")
    relevance_scores = []
    for image in test_dataset:
        similar_images = search_index(image)
        image_relevance = [1 for i in similar_images if i.id == image.id]
        relevance_scores.append(image_relevance)

    metrics = RankingMetrics(
        relevance=relevance_scores,
        n_relevant=config["n_relevant"],
        gains=config["gains"],
    )
    metrics_dict = metrics.compute_metrics(k=config["k"], metrics=[method])
    metrics_dict_avg = metrics.compute_average_metrics(k=config["k"], metrics=[method])

    return metrics_dict, metrics_dict_avg
