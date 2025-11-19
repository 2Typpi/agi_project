
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import geopandas as gpd
import io
import numpy as np
import math
import torch
from tqdm import trange
import warnings
import sys
from pathlib import Path

# Add parent directory to path to import RealTimeEnvironment
sys.path.append(str(Path(__file__).parent.parent))
from realtime_environment import RealTimeEnvironment

warnings.filterwarnings("ignore")

def generate_shape(poly, color, figsize, lim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0,lim])
    ax.set_ylim([0,lim])
    ax.axis('off')
    p = gpd.GeoSeries(poly)
    p.plot(ax=ax, color=color)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close(fig)
    return img_arr

# lim = 144
def plot_circle(r, lim, center=None):
    if center is None:
        center=(random.uniform(r,lim-r), random.uniform(r,lim-r))
    circle = Point(center).buffer(r)
    return circle

def plot_ragular_triangle(length, lim, left_down=None):
    if left_down is None:
        left_down = (random.uniform(0,lim-length), random.uniform(0,lim-length*(math.sqrt(3)/2)))
    triangle = Polygon([left_down, (left_down[0]+length, left_down[1]), (left_down[0]+length/2, left_down[1]+length*(math.sqrt(3)/2))])
    return triangle

def plot_square(length, lim, left_down=None):
    if left_down is None:
        left_down = (random.uniform(0,lim-length), random.uniform(0,lim-length))
    square = Polygon([left_down, (left_down[0]+length, left_down[1]), (left_down[0]+length, left_down[1]+length),  (left_down[0], left_down[1]+length)])
    return square
    
def plot_ragular_pentagon(r, lim, center=None):
    if center is None:
        center = (random.uniform(r*math.cos(math.pi/10),lim-r*math.cos(math.pi/10)), random.uniform(r*math.cos(math.pi/5), lim-r))
    p = []
    p.append((center[0]-r*math.sin(math.pi/5),center[1]-r*math.cos(math.pi/5)))
    p.append((center[0]+r*math.sin(math.pi/5),center[1]-r*math.cos(math.pi/5)))
    p.append((center[0]+r*math.cos(math.pi/10),center[1]+r*math.sin(math.pi/10)))
    p.append((center[0],center[1]+r))
    p.append((center[0]-r*math.cos(math.pi/10),center[1]+r*math.sin(math.pi/10)))
    pentagon = Polygon(p)
    return pentagon


class ShapeEnv(RealTimeEnvironment):
    def __init__(self, noisy=False, noise_value=0.1, device='cpu', time_scaling=1.0, return_states=False):
        # Initialize shape-specific attributes first
        self.color = ['r','g','b','k']
        self.shape = ['circle','triangle','square','pentagon']
        self.r = [15,30,45]
        self.color_count = 0
        self.shape_count = 0
        self.figsize=(2,2)
        self.lim = 144
        self.noisy = noisy
        self.noise_value = noise_value

        # Call parent constructor (which will call _get_initial_state)
        super().__init__(device=device, time_scaling=time_scaling, return_states=return_states)

    def generate_state(self):
        """Generate state as numpy array (internal helper)"""
        color = self.color[self.color_count % len(self.color)]
        shape = self.shape_count % (len(self.shape)*len(self.r))
        r = self.r[shape % len(self.r)]
        shape = self.shape[shape // len(self.r)]
        if shape == 'circle':
            state = generate_shape(plot_circle(r, self.lim),color,self.figsize,self.lim)
        if shape == 'triangle':
            state = generate_shape(plot_ragular_triangle(r*math.sqrt(2), self.lim),color,self.figsize,self.lim)
        if shape == 'square':
            state = generate_shape(plot_square(r*math.sqrt(3), self.lim),color,self.figsize,self.lim)
        if shape == 'pentagon':
            state = generate_shape(plot_ragular_pentagon(r, self.lim),color,self.figsize,self.lim)

        state = state/255
        if self.noisy:
            noise = np.random.normal(0, self.noise_value, state.shape)
            state = state + noise
            state = (state - np.min(state))/(np.max(state) - np.min(state))

        return state

    def _get_initial_state(self) -> torch.Tensor:
        """Get the initial state of the environment as a PyTorch tensor."""
        self.color_count = 0
        self.shape_count = 0
        state_np = self.generate_state()
        # Convert to torch tensor: shape (H, W, C) -> (C, H, W) for PyTorch convention
        state_tensor = torch.from_numpy(state_np).float()
        state_tensor = state_tensor.permute(2, 0, 1)  # HWC -> CHW
        return state_tensor.to(self.device)

    def _get_state(self) -> torch.Tensor:
        """Get current state of the environment."""
        return self.state.clone()

    def _step_simulation(self, action: torch.Tensor) -> torch.Tensor:
        """
        Perform one simulation step.

        Args:
            action: One-hot encoded action vector [5] where:
                - [1,0,0,0,0]: color-
                - [0,1,0,0,0]: color+
                - [0,0,1,0,0]: shape-
                - [0,0,0,1,0]: shape+
                - [0,0,0,0,1]: no-op (regenerate with new position/noise)

                If action is a scalar, it will be treated as the action index for backwards compatibility.

        Returns:
            observation: New observation after simulation step
        """
        # Handle both one-hot and scalar action formats
        if action.numel() == 1:
            # Scalar action (backwards compatibility)
            action_int = int(action.item())
        else:
            # One-hot encoded action
            action_int = torch.argmax(action).item()

        # Apply action
        if action_int == 0:
            self.color_count -= 1
        elif action_int == 1:
            self.color_count += 1
        elif action_int == 2:
            self.shape_count -= 1
        elif action_int == 3:
            self.shape_count += 1
        # else: no-op (action 4 or any other value)
        # Position/noise still changes because generate_state() uses random placement

        # Generate new state (position changes randomly even for no-op)
        state_np = self.generate_state()
        state_tensor = torch.from_numpy(state_np).float()
        state_tensor = state_tensor.permute(2, 0, 1)  # HWC -> CHW
        self.state = state_tensor.to(self.device)
        self.step_count += 1

        return self.state.clone()

    def render(self, figsize=(8, 8)):
        """
        Render the current state as an image.

        Args:
            figsize: Figure size for rendering

        Returns:
            numpy array of the rendered image
        """
        # Convert from CHW to HWC for visualization
        state_np = self.state.cpu().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(state_np)
        ax.axis('off')
        ax.set_title(f"Color: {self.color[self.color_count % len(self.color)]}, Step: {self.step_count}")
        return fig