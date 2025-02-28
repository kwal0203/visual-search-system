import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """Simple CNN for learning digit embeddings."""

    def __init__(self, embedding_dim=64):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Normalize embeddings to unit length
        x = F.normalize(x, p=2, dim=1)
        return x


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for metric learning."""

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        # x1, x2 are embeddings of the pairs
        # y is 1 for positive pairs, 0 for negative pairs
        distance = F.pairwise_distance(x1, x2)
        # Loss is different for positive and negative pairs
        positive_loss = y * distance.pow(2)
        negative_loss = (1 - y) * F.relu(self.margin - distance).pow(2)
        loss = (positive_loss + negative_loss).mean()
        return loss
