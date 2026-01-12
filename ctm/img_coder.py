from typing import Tuple
import torch.nn as nn
import torch

import torch.nn.functional as F

class ConvDecoder(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()

        # First project back to the spatial size used by the encoder's last conv layer
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # -> (3, 64, 64)
            nn.Sigmoid()  # optional: keeps output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 4, 4)
        x = self.deconv(x)
        return x
    
class ConvEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()

        # Input: (B, 3, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            # nn.Conv2d(128, 256, 4, stride=2, padding=1),# -> (256, 4, 4)
            # nn.ReLU(),
        )

        self.fc = nn.Linear(128 * 4 * 4, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ImageDecoder(nn.Module):
    def __init__(self, memory_dim, latent, cat):
        super(ImageDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim + latent * cat, 16*16),
            #Rearrange("b (c w h) -> b c w h", c = 16, w = 4, h = 4),
            nn.Upsample((16,16)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            nn.ReLU(),
            nn.Upsample((32,32)),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            nn.Sigmoid()
            # nn.Upsample((64,64)),
            # nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=(1,1), padding_mode="zeros"),
            # nn.Sigmoid()
 
        )
    def forward(self, x):
        # Assuming x is of shape (B, 3, W, H)
        x = self.net(x)
        return x
    
    def reconstruction_loss(self, state, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        decoded_img = self.forward(state.combined)
        # RSME loss:
        loss = torch.nn.MSELoss()(decoded_img, observation)
        return loss
    
class ImgDecoder(nn.Module):
    """
        Decodes stochastic and deterministic hidden state into a observation.
    """
    def __init__(self, memory_dim, latent_dim, categoricals) -> None:
        super(ImgDecoder, self).__init__()
        self.decoder = ConvDecoder(input_dim=latent_dim*categoricals + memory_dim)

    def forward(self, state_combined):
        img = self.decoder(state_combined)
        return img
    
    def reconstruction_loss(self, state, observation):
        """
            Computes reconstruction loss between decoded observation (from state) and given observation.
        """
        decoded_img = self.forward(state.combined)
        loss = torch.nn.MSELoss()(decoded_img, observation)
        return loss
