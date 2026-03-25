# preprocess.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------- DEFAULT CONFIG ----------------
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ---------------- DATA LOADER ----------------
def get_dataloaders(data_dir, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Loads training and validation datasets using ImageFolder.
    Automatically infers class names and number of classes.
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"'train' directory not found at: {train_dir}")

    # ImageNet normalization (MUST match training & inference)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }

    image_datasets = {}

    image_datasets["train"] = datasets.ImageFolder(
        root=train_dir,
        transform=data_transforms["train"]
    )

    if os.path.exists(val_dir):
        image_datasets["val"] = datasets.ImageFolder(
            root=val_dir,
            transform=data_transforms["val"]
        )
    else:
        print("Warning: 'val' directory not found. Validation will be skipped.")

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=4,
            pin_memory=True
        )
        for x in image_datasets
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

    # --------- CLASS INFORMATION ---------
    class_to_idx = image_datasets["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # --------- DATASET SUMMARY ---------
    print("\n--- Dataset Summary ---")
    print(f"Total Classes: {num_classes}")

    for class_name, idx in class_to_idx.items():
        count = sum(
            1 for _, label in image_datasets["train"].samples if label == idx
        )
        print(f"  [{idx}] {class_name}: {count} images")

    print(
        f"Train samples: {dataset_sizes.get('train', 0)}, "
        f"Val samples: {dataset_sizes.get('val', 0)}"
    )
    print("------------------------\n")

    return dataloaders, dataset_sizes, num_classes, idx_to_class
