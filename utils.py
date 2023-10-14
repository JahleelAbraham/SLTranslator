import torch
import random
import pandas as pd
import numpy as np

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())

def print_data(path):
    read = pd.read_csv(path).to_numpy()

    labels = list(map(extract_label, read))
    data = list(map(delete_label, read))

    for i in range(0, len(data)):
        print(f"Item #{i}- Label: {data[i][0]} - Data: e")


def delete_label(item):
    return np.delete(item, 0)


def extract_label(arr):
    return arr[0]

