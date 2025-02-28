from pathlib import Path
from torchvision import datasets, transforms
from PIL import Image as PILImage
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.models import Image, User, Base
import matplotlib.pyplot as plt


def setup_mnist_database(db_url: str = "sqlite:///mnist.db"):
    """Set up the database and load MNIST dataset."""
    # Create database
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # Create a system user for MNIST
    mnist_user = db.query(User).filter(User.username == "mnist_system").first()
    if not mnist_user:
        mnist_user = User(username="mnist_system", email="mnist@system.local")
        db.add(mnist_user)
        db.commit()
        db.refresh(mnist_user)

    return db, mnist_user


def load_mnist(save_dir: str = "data/mnist", db_url: str = "sqlite:///mnist.db"):
    """Download MNIST and store in database with file paths."""
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set up database
    db, mnist_user = setup_mnist_database(db_url)

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

    return "MNIST dataset loaded successfully"


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
            pairs.append((img1.image_id, img2.image_id))
            labels.append(0)

    print("100% - Pair generation complete!")

    # Visualize some random pairs
    def display_pair(img1_path, img2_path, is_same, ax):
        img1 = PILImage.open(img1_path)
        img2 = PILImage.open(img2_path)

        # Create a figure with two subplots side by side
        ax[0].imshow(img1, cmap="gray")
        ax[0].axis("off")
        ax[1].imshow(img2, cmap="gray")
        ax[1].axis("off")
        pair_type = "Same Digit" if is_same else "Different Digits"
        ax[0].set_title(f"{pair_type} Pair")

    # Create a figure with subplots for positive and negative pairs
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    plt.suptitle("Sample Contrastive Pairs", fontsize=14)

    # Show 2 positive and 2 negative pairs
    pos_pairs = [(p, l) for p, l in zip(pairs, labels) if l == 1]
    neg_pairs = [(p, l) for p, l in zip(pairs, labels) if l == 0]

    # Display 2 positive pairs
    for i in range(2):
        if pos_pairs:
            pair, _ = pos_pairs[i]
            img1 = db.query(Image).filter(Image.image_id == pair[0]).first()
            img2 = db.query(Image).filter(Image.image_id == pair[1]).first()
            display_pair(img1.file_path, img2.file_path, True, axes[i])

    # Display 2 negative pairs
    for i in range(2):
        if neg_pairs:
            pair, _ = neg_pairs[i]
            img1 = db.query(Image).filter(Image.image_id == pair[0]).first()
            img2 = db.query(Image).filter(Image.image_id == pair[1]).first()
            display_pair(img1.file_path, img2.file_path, False, axes[i + 2])

    plt.tight_layout()
    plt.show()

    return pairs, labels


if __name__ == "__main__":
    db, mnist_user = setup_mnist_database()
    load_mnist()
    pairs, labels = generate_contrastive_pairs(db)
    print(pairs)
    print(labels)
