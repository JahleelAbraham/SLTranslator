import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from hands import render
from dataset import ASLDatasetNoLabel
from model import NeuralNetwork
from torchvision import transforms
from utils import predict_by_max_logit
from torch.utils.data import DataLoader
from train import train

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
# you forgot Z? or maybe you dont need it idk lmao

# collect()
# train(10)

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # this maps pixels values from 0 to 255 to the 0 to 1 range and makes a PyTorch tensor
            transforms.Normalize((0.5,), (0.5,)),  # this then maps the pixel tensor values to the -1 to 1 range
        ]
    )

model = NeuralNetwork()
model.load_state_dict(torch.load("./models/SLT-ASLv1"))
model.eval()


def getModelResult(img):
    # FIXME: This currently crashes the model. More processing must be done
    sett = ASLDatasetNoLabel(np.array(img.reshape(28, 28), dtype=np.uint8), transform=transform)
    set_loader = DataLoader(sett)

    iterator = iter(set_loader)
    signs = next(iterator)

    with torch.no_grad():
        logits = model(signs[0])
        return alphabet[predict_by_max_logit(logits)]


while True:
    render(getModelResult)
