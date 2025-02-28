from pathlib import Path
from torchvision import datasets, transforms
from PIL import Image as PILImage
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.models import Image, User, Base
from src.data.utils import save_sample_pairs


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


def load_mnist(db, mnist_user, save_dir: str = "data/mnist"):
    """Download MNIST and store in database with file paths.

    Args:
        db: Database session
        mnist_user: User object for MNIST system
        save_dir: Directory to save MNIST images
    """
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

    # Save sample pairs using the utility function
    save_sample_pairs(pairs, labels, db, num_samples=2)

    return pairs, labels


if __name__ == "__main__":
    db, mnist_user = setup_mnist_database()
    load_mnist(db, mnist_user)
    pairs, labels = generate_contrastive_pairs(db)
    print(pairs)
    print(labels)
