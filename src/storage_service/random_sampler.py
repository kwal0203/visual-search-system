from torch.utils.data import Sampler

import random


class BalancedRandomPairBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.class_indices = dataset.class_indices

        # Ensure each class has at least 2 samples
        for cls, indices in self.class_indices.items():
            if len(indices) < 2:
                raise ValueError(f"Class {cls} has fewer than 2 samples!")

    def __iter__(self):
        values = self.class_indices.values()
        num_batches = min(len(indices) // 2 for indices in values)

        for _ in range(num_batches):
            batch = []
            for cls in self.classes:
                indices = self.class_indices[cls]

                # Randomly select two different samples
                idx1, idx2 = random.sample(indices, 2)
                batch.extend([idx1, idx2])

            yield batch

    def __len__(self):
        values = self.class_indices.values()
        return min(len(indices) // 2 for indices in values)
