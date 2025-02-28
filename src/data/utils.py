from pathlib import Path
from PIL import Image as PILImage
from src.data.models import Image


def save_sample_pairs(pairs, labels, db, num_samples=2):
    """
    Save sample pairs of MNIST digits to files.

    Args:
        pairs: List of tuples containing image ID pairs
        labels: List of labels (1 for same digit, 0 for different digits)
        db: Database session
        num_samples: Number of samples to save for each category (positive/negative)
    """
    # Save sample pairs to files
    output_dir = Path("sample_pairs")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories for positive and negative pairs
    pos_dir = output_dir / "positive_pairs"
    neg_dir = output_dir / "negative_pairs"
    pos_dir.mkdir(exist_ok=True)
    neg_dir.mkdir(exist_ok=True)

    def save_pair(img1_path, img2_path, pair_idx, is_positive=True):
        """Save a pair of images side by side"""
        img1 = PILImage.open(img1_path)
        img2 = PILImage.open(img2_path)

        # Create a new image with both digits side by side
        combined = PILImage.new("L", (img1.width * 2, img1.height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        # Save to appropriate directory
        save_dir = pos_dir if is_positive else neg_dir
        save_path = save_dir / f"pair_{pair_idx}.png"
        combined.save(save_path)
        return save_path

    # Split into positive and negative pairs
    pos_pairs = [(p, l) for p, l in zip(pairs, labels) if l == 1]
    neg_pairs = [(p, l) for p, l in zip(pairs, labels) if l == 0]

    print("\nSaving sample pairs to files...")

    # Save positive pairs
    for i in range(num_samples):
        if i < len(pos_pairs):
            pair, _ = pos_pairs[i]
            img1 = db.query(Image).filter(Image.image_id == pair[0]).first()
            img2 = db.query(Image).filter(Image.image_id == pair[1]).first()
            save_path = save_pair(img1.file_path, img2.file_path, i, is_positive=True)
            print(f"Saved positive pair {i+1} to {save_path}")

    # Save negative pairs
    for i in range(num_samples):
        if i < len(neg_pairs):
            pair, _ = neg_pairs[i]
            img1 = db.query(Image).filter(Image.image_id == pair[0]).first()
            img2 = db.query(Image).filter(Image.image_id == pair[1]).first()
            save_path = save_pair(img1.file_path, img2.file_path, i, is_positive=False)
            print(f"Saved negative pair {i+1} to {save_path}")

    print(f"\nSample pairs have been saved to the '{output_dir}' directory")
