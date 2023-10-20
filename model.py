from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=24)
        )

    def forward(self, x):
        logits = self.main(x)
        return logits
