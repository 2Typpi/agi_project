import torch.nn as nn

class UniversalActionHead(nn.Module):
    def __init__(self, latent_dim=256, num_actions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # x is the final thought from CTM: (Batch, 256)
        return self.net(x)