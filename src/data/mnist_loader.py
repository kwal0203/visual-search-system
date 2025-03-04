from src.data.models import Image, User, Base
from torchvision import datasets, transforms
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from PIL import Image as PILImage
from pathlib import Path

import numpy as np


def setup_mnist_database(db_path: str):
    """Set up the database and MNIST user"""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        mnist_user = db.query(User).filter(User.username == "mnist_system").first()
        if not mnist_user:
            mnist_user = User(username="mnist_system", email="mnist@system.local")
            db.add(mnist_user)
            db.commit()
    finally:
        db.close()


def load_mnist(db_path: str, save_dir: str = "data/mnist"):
    """Download MNIST and store in database with file paths.

    Args:
        db_path: Path to the SQLite database file
        save_dir: Directory to save MNIST images
    """
    # Create database session
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Get or create MNIST system user
        mnist_user = db.query(User).filter(User.username == "mnist_system").first()
        if not mnist_user:
            mnist_user = User(username="mnist_system", email="mnist@system.local")
            db.add(mnist_user)
            db.commit()

        # Create directories
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Download MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            "data", train=False, download=True, transform=transform
        )

        def process_dataset(dataset, split):
            for idx, (img_tensor, label) in enumerate(dataset):
                # Convert to PIL Image and save
                img = transforms.ToPILImage()(img_tensor)
                img_path = save_dir / f"{split}_{idx}.png"
                img.save(img_path)

                # Create database entry
                db_image = Image(
                    user_id=mnist_user.user_id,
                    file_path=str(img_path),
                    digit_label=int(label),
                    is_mnist=True,
                    dataset_split=split,
                    tags=f'{{"digit": {label}}}',
                )
                db.add(db_image)

                if idx % 1000 == 0:
                    db.commit()
                    print(f"Processed {idx} images from {split} set")

        # Process both splits
        process_dataset(train_dataset, "train")
        process_dataset(test_dataset, "test")
        db.commit()

        print("MNIST dataset loaded successfully")
    finally:
        # Always close the session when done
        db.close()


def generate_contrastive_pairs(db, num_pairs=10000, same_digit_ratio=0.5):
    """Generate positive and negative pairs for contrastive learning."""
    # Get all MNIST images from training set
    train_images = (
        db.query(Image)
        .filter(Image.is_mnist == True, Image.dataset_split == "train")
        .all()
    )

    pairs = []
    labels = []  # 1 for same digit, 0 for different digits

    print(f"Generating {num_pairs} contrastive pairs...")
    progress_step = num_pairs // 10  # Calculate step size for 10% intervals

    seen_positive = False
    seen_negative = False

    for i in range(num_pairs):
        if i > 0 and i % progress_step == 0:
            progress = (i / num_pairs) * 100
            print(f"Progress: {progress:.0f}% - Generated {i} pairs")

        if np.random.random() < same_digit_ratio:
            # Generate positive pair (same digit)
            digit = np.random.randint(0, 10)
            same_digit_images = [
                img for img in train_images if img.digit_label == digit
            ]
            if len(same_digit_images) < 2:
                continue
            pair = np.random.choice(same_digit_images, 2, replace=False)
            pairs.append((pair[0].image_id, pair[1].image_id))
            if not seen_positive:
                print(f"Positive pair: {pair[0].image_id} and {pair[1].image_id}")
                seen_positive = True
            labels.append(1)
        else:
            # Generate negative pair (different digits)
            digit1 = np.random.randint(0, 10)
            digit2 = (digit1 + np.random.randint(1, 10)) % 10
            digit1_images = [img for img in train_images if img.digit_label == digit1]
            digit2_images = [img for img in train_images if img.digit_label == digit2]
            if not digit1_images or not digit2_images:
                continue
            img1 = np.random.choice(digit1_images)
            img2 = np.random.choice(digit2_images)
            if not seen_negative:
                print(f"Negative pair: {img1.image_id} and {img2.image_id}")
                seen_negative = True
            pairs.append((img1.image_id, img2.image_id))
            labels.append(0)

    print("100% - Pair generation complete!")

    # Save sample pairs using the utility function
    save_sample_pairs(pairs, labels, db, num_samples=2)

    return pairs, labels


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
    output_dir = Path("training_outputs/sample_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for positive and negative pairs
    pos_dir = output_dir / "positive_pairs"
    neg_dir = output_dir / "negative_pairs"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    setup_mnist_database()
    load_mnist(db, mnist_user)
    pairs, labels = generate_contrastive_pairs(db)
    print(pairs)
    print(labels)
