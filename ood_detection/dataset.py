from typing import List

import torch
import numpy as np
import medmnist
import torchvision.transforms.v2 as transforms
from medmnist import INFO
from torch.utils.data import Dataset, DataLoader

from ood_detection import accelerator


def download(data_flag):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    DataClass.download(None)


def partial_dataset_wrapper(dataset: Dataset):
    class PartialDataset(dataset):
        def __init__(
            self, exclude_keys: List[str], label_dict: dict, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.exclude_keys = exclude_keys
            self.label_dict = label_dict
            if exclude_keys is not None:
                indice = np.ones_like(self.labels, dtype=bool)
                for key in exclude_keys:
                    indice &= self.labels != int(key)
                indice = indice.nonzero()[0]
                self.imgs = self.imgs[indice]
                self.labels = self.labels[indice]

                valid_keys = [
                    k for k in label_dict.keys() if k not in exclude_keys
                ]

                for i, k in enumerate(valid_keys):
                    self.labels[self.labels == int(k)] = i

                self.label_dict = {
                    i: label_dict[k] for i, k in enumerate(valid_keys)
                }

        @property
        def num_classes(self):
            return len(self.label_dict.keys())

    return PartialDataset


def prepare_dataset(
    data_flag: str = "pathmnist",
    split: str = "train",
    ood_keys: List[str] = None,
    download: bool = True,
) -> Dataset:
    ood_keys = [] if ood_keys is None else ood_keys
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info["python_class"])
    DataClass = partial_dataset_wrapper(DataClass)

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((384, 384), antialias=True),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    target_transform = lambda x: torch.tensor(x, dtype=torch.long).squeeze()

    dataset = DataClass(
        exclude_keys=ood_keys,
        label_dict=info["label"],
        split=split,
        transform=data_transform,
        target_transform=target_transform,
        download=download,
    )

    return dataset


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
) -> DataLoader:
    """获取DataLoader"""
    assert dataset.split in (
        "train",
        "val",
        "test",
    ), "Unrecognized dataset mode."
    return accelerator.prepare(
        DataLoader(
            dataset,
            batch_size,
            shuffle=dataset.split == "train",
        )
    )
