
from typing import Dict
import torch


class FunctionBuffer:
    """
    Creates a pseudo buffer from a .sample function.
    """

    def __init__(self, sample_function, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_function = sample_function

    def sample(self, batch_size, trajectory_len, to_device=None, keys=None, **kwargs) -> Dict:
        transitions = self.sample_function(batch_size, trajectory_len, to_device, keys, **kwargs)
        # transitions_filtered = {key: transitions[key].to(to_device) for key in keys}
        return transitions



class EpisodicBuffer(torch.nn.Module):
    """
    Episodic replay buffer, storing experiences from multiple episodes.
    Note that this buffer is saved when the parent nn.Module is saved!
    """

    def __init__(self, device, num_episodes=1000, max_episode_length=250, shapes={}):
        super().__init__()

        self.keys = []
        for key, value in shapes.items():
            self.keys.append(key)
            tensor = torch.zeros((num_episodes, max_episode_length, *value), device=device)
            self.register_buffer(key, tensor)

        idx = torch.zeros((num_episodes,), dtype=torch.int, device=device)
        episode_count = torch.zeros((num_episodes,), dtype=torch.int, device=device)

        self.register_buffer("episode_count", episode_count)
        self.register_buffer("idx", idx)
        self.register_buffer("episode", torch.tensor(0, device=device))
        self.register_buffer("max_episode", torch.tensor(0, device=device))
        self.register_buffer("episode_counter", torch.tensor(1, device=device))
        self.register_buffer("num_episodes", torch.tensor(num_episodes, device=device))
        self.register_buffer("max_episode_length", torch.tensor(max_episode_length, device=device))

        self.is_new_episode = False
        self.episode_count[0] = 1

    def append(self, data): # appends single data point to current episode
        self.is_new_episode = False

        episode = self.episode
        idx = self.idx[episode]
        for key, value in data.items():
            if hasattr(self, key):
                getattr(self, key)[episode, idx] = value

        self.idx[episode] = idx + 1

    def new_episode(self):
        self.is_new_episode = True
        self.episode = self.episode + 1
        
        if self.episode >= self.num_episodes:
            self.episode = torch.tensor(0, device=self.episode.device)
        else:
            self.max_episode = self.max_episode + 1  # keep counting

        self.idx[self.episode] = torch.tensor(
            0, device=self.idx.device
        )  # reset idx in episode
        self.episode_counter = self.episode_counter + 1
        self.episode_count[self.episode] = self.episode_counter

    def sample_current_episode(self, to_device=None, keys=None):
        if keys is None:
            keys = self.keys
        if to_device is None:
            to_device = self.idx.device
        episode = self.episode
 
        len = self.idx[episode]
        if len == 0:
            return None
        return_dict = {}
        for key in keys:
            return_dict[key] = getattr(self, key)[episode, :len].to(to_device).unsqueeze(0)
        return return_dict

    def sample(self, batch_size, trajectory_len, to_device=None, keys=None, unique=False, sample_end=False):
        """
        Sample multiple contiguous trajectories from the buffer. 
        If unique is set, the resulting batch_size may be smaller than the desired batch_size, if not enough data is available.
        """

        if keys is None:
            keys = self.keys

        if to_device is None:
            to_device = self.idx.device

        available_episode = torch.arange(0, self.idx.shape[0], device=self.idx.device)[
            self.idx >= trajectory_len
        ]
        if (
            available_episode.size(0) == 0
        ):  # No episode can handle the desired trajectory length
            return None

        n_available = available_episode.shape[0]

        # Sample unique if possible
        if n_available >= batch_size:
            sel = torch.randperm(n_available, device=self.idx.device)[:batch_size]
        else: # not possible
            if unique: # enforce uniqueness by reducing the batch size
                sel = torch.randperm(n_available, device=self.idx.device)
            else: # fill up with non unique samples
                sel = torch.cat([torch.randperm(n_available, device=self.idx.device), 
                                torch.randint(0, n_available, (batch_size - n_available,), device=self.idx.device)])


        batch_size = sel.shape[0]
        episodes = available_episode[sel]
        if not sample_end:
            start = (
                (
                    torch.rand((batch_size,), device=self.idx.device)
                    * (self.idx[episodes] - trajectory_len + 1)
                )
                .int()
                .reshape((batch_size, 1))
            )
        else:
            start = (self.idx[episodes]-trajectory_len).reshape((batch_size, 1))

        episodes = episodes.reshape((-1, 1))
        indices = start + torch.arange(
            0, trajectory_len, device=self.idx.device
        ).reshape((1, trajectory_len))

        return_dict = {}


        all_indices = torch.arange(self.max_episode_length * self.num_episodes, device=self.idx.device).reshape(
            self.num_episodes, self.max_episode_length
        )

        flattened_indices = all_indices[episodes, indices]

        return_dict["episodes"] = episodes.to(to_device)
        return_dict["flattened_indices"] = flattened_indices.to(to_device)
        for key in keys:
            return_dict[key] = getattr(self, key)[episodes, indices].to(to_device)
        return return_dict

    def flatten(self, to_device, keys=None):
        """
        Get all available data as one dictionary containing a flattened tensor for all attributes.
        """
        if to_device is None:
            to_device = self.idx.device

        if keys is None:
            keys = self.keys

        mask = (
            torch.arange(self.max_episode_length, device=self.idx.device).unsqueeze(0) < self.idx.unsqueeze(1)
        ).unsqueeze(-1)

        return_dict = {}
        for key in keys:
            data = getattr(self, key)
            return_dict[key] = (
                data[mask.expand_as(data)].view(-1, data.shape[-1]).to(to_device)
            )
        return return_dict
    def clear(self):
        """
        Clear all data in the buffer.
        """
        for key in self.keys:
            tensor = getattr(self, key)
            tensor.zero_()
        self.idx.zero_()
        self.episode.zero_()
        self.max_episode.zero_()
        self.episode_counter.fill_(1)
        self.episode_count.zero_()
        self.episode_count[0] = 1