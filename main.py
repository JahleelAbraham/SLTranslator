import matplotlib.pyplot as plt
import numpy as np
import torch

from time import time
from random import randint
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ASLDataset
from torchvision import transforms
from utils import print_data, set_seeds, predict_by_max_logit, compute_accuracy_from_predictions
from plot import make_loss_plot, view_classify

# print_data("./training/data.csv")  # Print all the training data formatted

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

set_seeds(seed=randint(0, 9999))  # make learning repeatable by setting seeds

# number of examples to process at one time, the training set (50,000) is too big to do them all at once
batch_size = 44

transform = transforms.Compose(
   [
        transforms.ToTensor(),  # this maps pixels values from 0 to 255 to the 0 to 1 range and makes a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)),  # this then maps the pixel tensor values to the -1 to 1 range
   ]
)


train_set = ASLDataset("./training/data.csv", transform=transform)
test_set = ASLDataset("./testing/data.csv", transform=transform)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

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
num_of_images = 40
for index in range(1, num_of_images + 1):
    plt.subplot(4, 10, index).set_title(f"{alphabet[labels[index]]}")
    plt.axis('off')
    plt.imshow(signs[index].numpy().squeeze(), cmap='gray')

plt.show()

# define the model
input_size = 784  # 28 x 28 pixels, flattened
hidden_sizes = [128, 64]  # sizes of the 2 hidden layers
output_size = 26  # output size - one for each digit

# TODO: Increase the accuracy of the model. Right now its about %4.30
# define the network - 3 linear layers with ReLU activation functions
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(in_features=256, out_features=output_size)
)


print("Number of model parameters = {}".format(sum(p.numel() for p in model.parameters())))

# loss is cross entropy loss
loss_fn = nn.CrossEntropyLoss()

# the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# train the model
time_start = time()  # set a timer
epochs = 10  # number of training epochs
training_losses = []
for e in range(epochs):
    epoch_losses = []
    for signs, labels in train_loader:
        logits = model(signs)  # pass the features forward through the model
        loss = loss_fn(logits, labels)   # compute the loss
        epoch_losses.append(loss)

        optimizer.zero_grad()   # clear the gradients
        loss.backward()  # compute the gradients via backpropagation
        optimizer.step()  # update the weights using the gradients

    epoch_loss = np.array(torch.hstack(epoch_losses).detach().numpy()).mean()
    training_losses.append(epoch_loss)
    print("Epoch {} - Loss: {}".format(e + 1, epoch_loss))

print("\nTraining Time (in minutes) = {}".format((time() - time_start) / 60))


# plot the loss vs epoch
make_loss_plot(epochs, training_losses)


# get the first batch of test examples, so we can examine them
iterator = iter(test_loader)
signs, labels = next(iterator)

# display an image with the probability that it is correct
# Turn off gradients to speed up this part
with torch.no_grad():
    prob = torch.softmax(model(signs[0].unsqueeze(dim=0)), dim=-1)

# Output of the network are log-probabilities, need to take exponential for probabilities
probability = list(prob.numpy()[0])
print("Predicted Digit =", probability.index(max(probability)))
view_classify(signs[0].view(1, 28, 28), prob, alphabet)

# compute accuracy on the test set
predictions = []
labels_test = []
with torch.no_grad():  # don't need gradients for testing
    for images, labels in test_loader:
        labels_test.append(labels)
        with torch.no_grad():
            logits = model(signs)
            predictions.append(predict_by_max_logit(logits))  # make prediction on the class that has the highest value

print("Accuracy = {0:0.1f}%"
      .format(compute_accuracy_from_predictions(torch.hstack(predictions), torch.hstack(labels_test)) * 100.0))
