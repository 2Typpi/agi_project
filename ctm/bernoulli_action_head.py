import torch.nn as nn

class BernoulliActionHead(nn.Module):
    def __init__(self, latent_dim=256, num_actions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # x is the final thought from CTM: (Batch, latent_dim)
        # Returns logits for Bernoulli distribution: (Batch, num_actions)
        return self.net(x)
