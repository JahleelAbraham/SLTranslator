import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class ASLDataset(Dataset):
    def __init__(self, dataCsv, transform=None, target_transform=None):
        self.dataCsv = dataCsv
        self.transform = transform
        self.target_transform = target_transform
