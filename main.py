import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from random import randint
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ASLDataset
from torchvision import transforms
from utils import print_data, set_seeds

# print_data("./training/data.csv")  # Print all the training data formatted


set_seeds(seed=randint(0, 9999))  # make learning repeatable by setting seeds

batch_size = 128  # number of examples to process at one time, the training set (50,000) is too big to do them all at once

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # this maps pixels values from 0 to 255 to the 0 to 1 range and makes a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)),  # this then maps the pixel tensor values to the -1 to 1 range
    ]
)

train_set = ASLDataset("./training/data.csv", transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

iterator = iter(train_loader)
signs, labels = next(iterator)

print(signs.shape)
print(labels.shape)

img = signs[0].squeeze()
label = labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(signs[index].numpy().squeeze(), cmap='gray')

plt.show()

# TODO: Fix training the model. The training simply returns the below error
# FIXME: RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
# # define the model
# input_size = 784  # 28 x 28 pixels, flattened
# hidden_sizes = [128, 64]  # sizes of the 2 hidden layers
# output_size = 10  # output size - one for each digit
#
# # define the network - 3 linear layers with ReLU activation functions
# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(input_size, hidden_sizes[0]),
#     nn.ReLU(),
#     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#     nn.ReLU(),
#     nn.Linear(hidden_sizes[1], output_size)
# )
#
#
# print("Number of model parameters = {}".format(sum(p.numel() for p in model.parameters())))
#
# # loss is cross entropy loss
# loss_fn = nn.CrossEntropyLoss()
#
# # the optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
#
# # train the model
# time_start = time()  # set a timer
# epochs = 100  # number of training epochs
# training_losses = []
# for e in range(epochs):
#     epoch_losses = []
#     for images, labels in train_loader:
#         logits = model(signs)  # pass the features forward through the model
#         loss = loss_fn(logits, labels)   # compute the loss
#         epoch_losses.append(loss)
#
#         optimizer.zero_grad()   # clear the gradients
#         loss.backward()  # compute the gradients via backpropagation
#         optimizer.step()  # update the weights using the gradients
#
#     epoch_loss = np.array(torch.hstack(epoch_losses).detach().numpy()).mean()
#     training_losses.append(epoch_loss)
#     print("Epoch {} - Loss: {}".format(e, epoch_loss))
#
# print("\nTraining Time (in minutes) = {}".format((time() - time_start) / 60 ))


