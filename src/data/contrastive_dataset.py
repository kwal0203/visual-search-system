from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.models import Image as ImageModel

import numpy as np


class ContrastivePairDatasetMNIST(Dataset):
    def __init__(self, db_path, dataset_split="train", transform=None):
        self.db_path = db_path
        self.dataset_split = dataset_split
        self.transform = transform

        # Connect to DB and retrieve all filepaths and labels
        engine = create_engine(f"sqlite:///{db_path}")
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        try:
            # Query images using SQLAlchemy ORM
            images = (
                db.query(ImageModel)
                .filter(ImageModel.dataset_split == self.dataset_split)
                .filter(ImageModel.is_mnist == True)
                .all()
            )

            # Extract filepaths and labels
            self.filepaths = [img.file_path for img in images]
            self.targets = np.array([img.digit_label for img in images])

            # Identify unique classes
            self.classes = np.unique(self.targets)
            print(self.classes)

            # Create mapping: class -> list of indices
            self.class_indices = {}
            for cls in self.classes:
                self.class_indices[cls] = np.where(self.targets == cls)[0].tolist()
        finally:
            db.close()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.filepaths[idx], self.targets[idx]
        image = Image.open(filepath).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label
