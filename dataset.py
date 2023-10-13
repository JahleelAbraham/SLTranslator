import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class ASLDataset(Dataset): # TODO: Create dataset type
    def __init__(self, dataCsv, transform=None, target_transform=None):
        self.dataCsv = pd.read_csv(dataCsv)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataCsv.rows)

    def __getitem__(self, i):
        return self.dataCsv.rows[i+1][self.dataCsv.columns > 0]
