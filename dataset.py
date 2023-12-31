import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def delete_label(item):
    return np.delete(item, 0)


def extract_label(arr):
    label = arr[0]
    if label > 9:
        return label - 1
    return label


class ASLDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        read = pd.read_csv(path).to_numpy()

        self.data = list(map(delete_label, read))
        self.labels = list(map(extract_label, read))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        transformed = Image.fromarray(np.array(self.data[i].reshape(28, 28), dtype=np.uint8))

        if self.transform is not None:
            transformed = self.transform(transformed)

        return transformed, self.labels[i]


class ASLDatasetNoLabel(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, i):
        transformed = Image.fromarray(self.data)

        if self.transform is not None:
            transformed = self.transform(transformed)

        return [transformed]
