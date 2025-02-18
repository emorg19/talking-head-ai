# model.py - Defines Generator and Discriminator
import torch.nn as nn

# Define Generator Network
class TalkingHeadGenerator(nn.Module):
    def __init__(self, input_dim=100, output_channels=3):
        super(TalkingHeadGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels * 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 3, 64, 64)

# Define Discriminator Network
class TalkingHeadDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(TalkingHeadDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels * 64 * 64, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)