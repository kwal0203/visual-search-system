from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import sqlite3


class ContrastivePairDatasetMNIST(Dataset):
    def __init__(self, db_path, dataset_split="train", transform=None):
        self.db_path = db_path
        self.dataset_split = dataset_split
        self.transform = transform

        # Connect to DB and retrieve all filepaths and labels
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT file_path, digit_label FROM images
            WHERE dataset_split = ? AND is_mnist = 1
        """,
            (self.dataset_split,),
        )

        self.data = cursor.fetchall()
        conn.close()

        # Extract filepaths and labels
        self.filepaths, self.targets = zip(*self.data)
        self.targets = np.array(self.targets)

        # Identify unique classes
        self.classes = np.unique(self.targets)
        print(self.classes)

        # Create mapping: class -> list of indices
        self.class_indices = {}
        for cls in self.classes:
            self.class_indices[cls] = np.where(self.targets == cls)[0].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.filepaths[idx], self.targets[idx]
        image = Image.open(filepath).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label
