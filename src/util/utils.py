from torchvision import transforms
from typing import Optional
from pathlib import Path

import json


def load_training_config(config_path: Optional[str] = None) -> dict:
    """Load training configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file. If None, uses default config.

    Returns:
        Dictionary containing training configuration.
    """
    config_path = (
        Path(config_path) if config_path else Path(__file__).parent / "config.json"
    )

    try:
        with config_path.open("r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    return config


def get_transform(name: str):
    if name == "mnist":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        raise ValueError(f"Transform {name} not found")
