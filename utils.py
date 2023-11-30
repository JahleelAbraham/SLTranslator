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
    print(logits)
    return torch.argmax(logits, dim=-1)


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())

def yesno(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' " "(or 'y' or 'n')")

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

