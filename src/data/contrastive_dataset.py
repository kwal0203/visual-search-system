import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage

from src.data.models import Image


class ContrastivePairDataset(Dataset):
    """Dataset for training with contrastive pairs."""

    def __init__(self, pairs, labels, db):
        self.pairs = pairs
        self.labels = labels
        self.db = db
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]

        # Get images from database
        img1 = self.db.query(Image).filter(Image.image_id == pair[0]).first()
        img2 = self.db.query(Image).filter(Image.image_id == pair[1]).first()

        # Load and transform images
        img1 = PILImage.open(img1.file_path)
        img2 = PILImage.open(img2.file_path)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.FloatTensor([label])
