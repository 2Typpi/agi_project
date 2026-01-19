import torch.nn as nn

class UniversalCriticHead(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output: The Value V(s)
        )

    def forward(self, final_thought):
        return self.net(final_thought)