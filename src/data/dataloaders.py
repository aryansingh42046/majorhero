
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import RetinalDataset


# Project root
ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT / "datasets" / "manifests" / "final_dataset_splits.csv"
)


def get_transforms(split):
    """
    Training gets augmentation.
    Validation and testing get no random augmentation.
    """

    if split == "train":

        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])

    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


def create_dataloaders(
    dataset_name,
    batch_size=4,
    num_workers=0,
    manifest_path=MANIFEST_PATH,
):
    """
    Creates train, validation and test DataLoaders
    for one retinal dataset.

    Example:
        loaders = create_dataloaders("ACRIMA")
        train_loader = loaders["train"]
    """

    dataset_name = dataset_name.upper()

    valid_datasets = ["ACRIMA", "IDRID", "MDD", "ODIR"]

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset: {dataset_name}"
        )

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loaders = {}

    for split in ["train", "validation", "test"]:

        dataset = RetinalDataset(
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            split=split,
            transform=get_transforms(split),
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    print(f"\nDataLoaders created for {dataset_name}")
    print("Device:", device)
    print("Batch size:", batch_size)

    return loaders
