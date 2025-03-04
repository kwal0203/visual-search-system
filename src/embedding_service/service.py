from src.embedding_service.train import train_embedding_model
from src.embedding_service.models import get_embedding_model
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from src.embedding_service.contrastive_dataset import ContrastivePairDatasetMNIST
from src.embedding_service.random_sampler import BalancedRandomPairBatchSampler

import torch


def get_dataloader(db_path: str, transform: transforms.Compose):
    train_dataset = ContrastivePairDatasetMNIST(
        db_path=db_path,
        dataset_split="train",
        transform=transform,
    )
    test_dataset = ContrastivePairDatasetMNIST(
        db_path=db_path,
        dataset_split="test",
        transform=transform,
    )

    train_sampler = BalancedRandomPairBatchSampler(train_dataset)
    test_sampler = BalancedRandomPairBatchSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    return train_dataloader, test_dataloader


def embed_image(
    image_path: str, model_path: str, transform: transforms.Compose
) -> torch.Tensor:
    """API ENDPOINT"""
    embedding_model = torch.load(model_path)
    image = Image.open(image_path)
    image = transform(image)
    # TODO: Send the new image to object store service
    # TODO: Send the image to the nearest neighbor search service
    return embedding_model.embed_image(image)


def train_model(
    db_path: str,
    config_path: str,
    model_path: str,
):
    """API ENDPOINT"""
    if not Path(model_path).exists():
        print("Training embedding model...")
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataloader, test_dataloader = get_dataloader(
            db_path=db_path, transform=transform
        )

        return train_embedding_model(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            config_path=config_path,
        )

    print("Loading existing model...")
    return torch.load(model_path)


# train_dataloader, test_dataloader = get_dataloader(
#     db_path=db_path, save_dir=save_dir
# )

# # Check if model exists, if not train it
# model_path = (
#     PROJECT_ROOT / "src" / "embedding_service" / "model" / "embedding_model.pth"
# )
# model_path.parent.mkdir(
#     exist_ok=True
# )  # Create models directory if it doesn't exist

# if not model_path.exists():
#     print("Training embedding model...")
#     config_path = PROJECT_ROOT / "src" / "embedding_service" / "config.json"
#     print(f"Using config from: {config_path}")
#     model = train_embedding_model(
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         config_path=str(config_path),
#     )
# else:
#     print("Loading existing model...")
#     model = torch.load(model_path)
#     model = model.to(device)
