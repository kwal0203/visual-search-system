from src.evaluation_service.metrics import RankingMetrics
from src.storage_service.service import get_test_dataset
from src.search_service.service import search_index
from src.util.utils import load_config
from pathlib import Path
from typing import List

CONFIG_PATH = Path(__file__).parent / "config.json"


def evaluate(method: List = None):
    config = load_config(CONFIG_PATH)
    test_dataset = get_test_dataset(dataset_name="mnist")
    relevance_scores = []
    idx = 0
    bank = 0
    for image in test_dataset:
        similar_images = search_index(image)
        image_relevance = []
        for i in similar_images:
            if i["digit"] == image.digit_label:
                image_relevance.append(1)
            else:
                image_relevance.append(0)
        relevance_scores.append(image_relevance)
        x = sum(image_relevance)
        if x == 0:
            bank += 1
        idx += 1

    metrics = RankingMetrics(
        relevance=relevance_scores,
        n_relevant=config["n_relevant"],
        gains=config["gains"],
    )
    # metrics_dict = metrics.compute_metrics(k=config["k"], metrics=method)
    metrics_dict_avg = metrics.compute_average_metrics(k=config["k"], metrics=method)

    return metrics_dict_avg
