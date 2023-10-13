import pandas as pd
import numpy as np

read = pd.read_csv("./training/data.csv").to_numpy()


def delete_label(item):
    return np.delete(item, 0)


def extract_label(arr):
    return arr[0]


data = list(map(delete_label, read))
labels = list(map(extract_label, read))

for i in range(0, len(data)):
    print(f"Item #{i}- Label: {data[i][0]} - Data: {np.delete(data[i], 0)}")
