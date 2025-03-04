from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing."""

    @abstractmethod
    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess a single image."""
        pass

    def preprocess_batch(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self.preprocess(img) for img in images])


class StandardImagePreprocessor(ImagePreprocessor):
    """Standard image preprocessing pipeline with configurable parameters."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        color_mode: str = "RGB",
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.color_mode = color_mode

        # Build transform pipeline
        transforms = [
            T.Resize(image_size),
            T.ToTensor(),
        ]

        if normalize:
            transforms.append(T.Normalize(mean=mean, std=std))

        self.transform = T.Compose(transforms)

    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess a single image."""
        if isinstance(image, str):
            image = Image.open(image)

        if image.mode != self.color_mode:
            image = image.convert(self.color_mode)

        return self.transform(image)


class AugmentedImagePreprocessor(StandardImagePreprocessor):
    """Image preprocessor with data augmentation for training."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        color_mode: str = "RGB",
        random_crop: bool = True,
        random_flip: bool = True,
        color_jitter: bool = True,
    ):
        super().__init__(image_size, mean, std, normalize, color_mode)

        # Build augmentation pipeline
        transforms = []

        if random_crop:
            transforms.extend(
                [
                    T.RandomResizedCrop(image_size),
                ]
            )
        else:
            transforms.extend(
                [
                    T.Resize(image_size),
                ]
            )

        if random_flip:
            transforms.append(T.RandomHorizontalFlip())

        if color_jitter:
            transforms.append(
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            )

        transforms.extend(
            [
                T.ToTensor(),
            ]
        )

        if normalize:
            transforms.append(T.Normalize(mean=mean, std=std))

        self.transform = T.Compose(transforms)


def get_preprocessor(
    preprocessor_type: str = "standard", **kwargs
) -> ImagePreprocessor:
    """Factory function to create image preprocessors."""
    preprocessors = {
        "standard": StandardImagePreprocessor,
        "augmented": AugmentedImagePreprocessor,
    }

    if preprocessor_type not in preprocessors:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")

    return preprocessors[preprocessor_type](**kwargs)
