
import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ODIR_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]

LABEL_MAPPINGS = {
    "ACRIMA": {
        "glaucoma": 1,
        "non_glaucoma": 0,
    },
    "MDD": {
        "amd": 0,
        "cataract": 1,
    },
    "IDRID": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    },
}


class RetinalDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        dataset_name,
        split,
        transform=None,
    ):

        self.dataset_name = dataset_name.upper()
        self.split = split.lower()

        valid_datasets = ["ACRIMA", "IDRID", "MDD", "ODIR"]
        valid_splits = ["train", "validation", "test"]

        if self.dataset_name not in valid_datasets:
            raise ValueError(f"Invalid dataset: {dataset_name}")

        if self.split not in valid_splits:
            raise ValueError(f"Invalid split: {split}")

        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)

        df = pd.read_csv(self.manifest_path)

        self.data = df[
            (df["dataset"].astype(str).str.upper() == self.dataset_name)
            & (df["split"].astype(str).str.lower() == self.split)
        ].copy().reset_index(drop=True)
        
        if self.data.empty:
            raise ValueError(
                f"No records found for {self.dataset_name}/{self.split}"
            )

        self.transform = transform or transforms.ToTensor()

        missing = [
            path for path in self.data["output_path"]
            if not Path(path).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images missing. First: {missing[0]}"
            )

        print(
            f"{self.dataset_name} | {self.split} | "
            f"Images: {len(self.data)}"
        )

    def __len__(self):
        return len(self.data)

    def encode_label(self, raw_label):

        if self.dataset_name == "ODIR":

            labels = (
                raw_label
                if isinstance(raw_label, list)
                else ast.literal_eval(str(raw_label))
            )

            target = torch.zeros(
                len(ODIR_LABELS),
                dtype=torch.float32
            )

            for label in labels:
                if label not in ODIR_LABELS:
                    raise ValueError(f"Unknown ODIR label: {label}")

                target[ODIR_LABELS.index(label)] = 1.0

            return target

        if self.dataset_name == "IDRID":
            label = str(int(float(raw_label)))
        else:
            label = str(raw_label).strip().lower()

        mapping = LABEL_MAPPINGS[self.dataset_name]

        if label not in mapping:
            raise ValueError(f"Unknown label: {label}")

        return torch.tensor(mapping[label], dtype=torch.long)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["output_path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        label = self.encode_label(row["label"])

        return {
            "image": image,
            "label": label,
            "dataset": self.dataset_name,
            "filename": row["filename"],
        }
