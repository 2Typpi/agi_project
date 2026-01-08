from torch.distributions import OneHotCategoricalStraightThrough
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class BernoulliStraightThrough(Bernoulli):
    """
    Custom Bernoulli distribution with modified log_prob and rsample methods.
    - log_prob: returns mean over the final dimension
    - rsample: uses reparameterization trick (straight-through estimator)
    """
    
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
    
    def log_prob(self, value):
        """
        Compute log probability and return mean over the final dimension.
        """
        log_probs = super().log_prob(torch.round(value))
        return log_probs.mean(dim=-1)
    
    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling using the trick: sample + probs - probs.detach()
        This maintains gradients through probs while keeping the distribution correct.
        """
        shape = self._extended_shape(sample_shape)
        probs = self.probs  # Use the property, not _probs
        
        # Sample from uniform distribution
        uniforms = torch.rand(shape, dtype=probs.dtype, device=probs.device)
        samples = (uniforms < probs).float()
        samples = samples + probs - probs.detach()
        return samples


class RandomBernoulliPolicy(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape
    
    def get_dist(self, obs, goal, horizon=None):
        """Get uniform random distribution"""
        logits = torch.zeros((*obs.shape[:-1], self.action_shape), device=obs.device)
        dist = BernoulliStraightThrough(logits=logits)
        return dist
    
    def get_action(self, obs, goal, horizon=None):
        """Sample action from policy"""
        dist = self.get_dist(obs, goal, horizon)
        return dist.rsample()
    
    def get_prior(self, dist):
        """Get uniform prior distribution"""
        prior = BernoulliStraightThrough(logits=torch.ones_like(dist.logits))
        return prior
    
class BernoulliPolicy(nn.Module):
    def __init__(self, obs_shape, goal_shape, action_shape, size_h=256, num_h=2, horizon=False):
        super().__init__()
        
        # Build simple feedforward network
        layers = []
        current_size = obs_shape + goal_shape + (1 if horizon else 0)
        
        # Hidden layers
        for _ in range(num_h):
            layers.extend([
                nn.Linear(current_size, size_h),
                nn.ReLU()
            ])
            current_size = size_h
        
        # Output layer
        layers.append(nn.Linear(current_size, action_shape))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs, goal, horizon=None):
        """Forward pass combining obs and goal"""
        if horizon is not None:
            input_tensor = torch.cat([obs, goal, horizon], dim=-1)
        else:   
            input_tensor = torch.cat([obs, goal], dim=-1)
        return self.net(input_tensor)
    
    def get_dist(self, obs, goal, horizon=None):
        """Get distribution from observations and goal"""
        logits = self.forward(obs, goal, horizon)
        dist = BernoulliStraightThrough(logits=logits)
        return dist
    
    def get_action(self, obs, goal, horizon=None):
        """Sample action from policy"""
        dist = self.get_dist(obs, goal, horizon)
        return dist.rsample() #+ dist.probs - dist.probs.detach()
    
    def get_prior(self, dist):
        """Get uniform prior distribution"""
        prior = BernoulliStraightThrough(logits=torch.ones_like(dist.logits))
        return prior
    

    

class DiscretePolicy(nn.Module):
    def __init__(self, obs_shape, goal_shape, action_shape, size_h=256, num_h=2):
        super().__init__()
        
        # Build simple feedforward network
        layers = []
        current_size = obs_shape + goal_shape
        
        # Hidden layers
        for _ in range(num_h):
            layers.extend([
                nn.Linear(current_size, size_h),
                nn.ReLU()
            ])
            current_size = size_h
        
        # Output layer
        layers.append(nn.Linear(current_size, action_shape))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs, goal):
        """Forward pass combining obs and goal"""
        input_tensor = torch.cat([obs, goal], dim=-1)
        return self.net(input_tensor)
    
    def get_dist(self, obs, goal):
        """Get distribution from observations and goal"""
        logits = self.forward(obs, goal)
        dist = OneHotCategoricalStraightThrough(logits=logits)
        return dist
    
    def get_action(self, obs, goal):
        """Sample action from policy"""
        dist = self.get_dist(obs, goal)
        return dist.rsample()
    
    def get_prior(self, dist):
        """Get uniform prior distribution"""
        prior = OneHotCategoricalStraightThrough(logits=torch.ones_like(dist.logits))
        return prior
    
class RandomDiscretePolicy(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape
    
    def get_dist(self, obs, goal):
        """Get uniform random distribution"""
        logits = torch.ones((obs.shape[0], self.action_shape), device=obs.device)
        dist = OneHotCategoricalStraightThrough(logits=logits)
        return dist
    
    def get_action(self, obs, goal):
        """Sample action from policy"""
        dist = self.get_dist(obs, goal)
        return dist.rsample()
    
    def get_prior(self, dist):
        """Get uniform prior distribution"""
        prior = OneHotCategoricalStraightThrough(logits=torch.ones_like(dist.logits))
        return prior