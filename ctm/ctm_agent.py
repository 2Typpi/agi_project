import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

from ctm.action_head import UniversalActionHead
from ctm.bernoulli_action_head import BernoulliActionHead
from ctm.critic_head import UniversalCriticHead

class CTMAgent(nn.Module):
    def __init__(self, ctm, continuous_state_trace, device, num_actions, action_type='categorical'):
        super().__init__()

        self.continious_state_trace = continuous_state_trace
        self.device = device
        self.action_type = action_type

        self.ctm = ctm
        actor_input_dim = critic_input_dim = self.ctm.synch_representation_size_out
        print(actor_input_dim, critic_input_dim)

        if action_type == 'bernoulli':
            self.actor = BernoulliActionHead(latent_dim=actor_input_dim, num_actions=num_actions)
        else:
            self.actor = UniversalActionHead(latent_dim=actor_input_dim, num_actions=num_actions)
        self.critic = UniversalCriticHead(latent_dim=critic_input_dim)


    def get_initial_state(self, num_envs):
        return self.get_initial_ctm_state(num_envs)

    def get_initial_ctm_state(self, num_envs):
        initial_state_trace = torch.repeat_interleave(self.ctm.start_trace.unsqueeze(0), num_envs, 0)
        initial_activated_state_trace = torch.repeat_interleave(self.ctm.start_activated_trace.unsqueeze(0), num_envs, 0)
        return initial_state_trace, initial_activated_state_trace
    
    def get_initial_lstm_state(self, num_envs):
        initial_hidden_state = torch.repeat_interleave(self.ctm.start_hidden_state.unsqueeze(0), num_envs, 0)
        initial_cell_state = torch.repeat_interleave(self.ctm.start_cell_state.unsqueeze(0), num_envs, 0)
        return initial_hidden_state, initial_cell_state

    def _get_hidden_states(self, state, done, num_envs):
        return self._get_ctm_hidden_states(state, done, num_envs)

    def _get_ctm_hidden_states(self, ctm_state, done, num_envs):
        initial_state_trace, initial_activated_state_trace = self.get_initial_ctm_state(num_envs)
        if self.continious_state_trace:
            masked_previous_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[0]
            masked_previous_activated_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[1]
            masked_initial_state_trace = done.view(-1, 1, 1) * initial_state_trace
            masked_initial_activated_state_trace = done.view(-1, 1, 1) * initial_activated_state_trace
            return (masked_previous_state_trace + masked_initial_state_trace), (masked_previous_activated_state_trace + masked_initial_activated_state_trace)
        else:
            return (initial_state_trace, initial_activated_state_trace)

    def get_states(self, x, ctm_state, done, track=False):
        num_envs = ctm_state[0].shape[0]

        if len(x.shape) == 4:
            _, C, H, W = x.shape
            xs = x.reshape((-1, num_envs, C, H, W))
        elif len(x.shape) == 2:
            _, C = x.shape
            xs = x.reshape((-1, num_envs, C))
        else:
            raise ValueError("Input shape not supported.")
        
        done = done.reshape((-1, num_envs))
        new_hidden = []
        for x, d in zip(xs, done):
            if not track:
                synchronisation, ctm_state = self.ctm(x, self._get_hidden_states(ctm_state, d, num_envs))
                tracking_data = None
                new_hidden += [synchronisation]
            else:
                synchronisation, ctm_state, pre_activations, post_activations = self.ctm(x, self._get_hidden_states(ctm_state, d, num_envs), track=True)
                tracking_data = {
                    'pre_activations': pre_activations,
                    'post_activations': post_activations,
                    'synchronisation': synchronisation.detach().cpu().numpy(),
                }
                new_hidden += [synchronisation]
        
        return torch.cat(new_hidden), ctm_state, tracking_data

    def get_value(self, x, ctm_state, done):
        hidden, _, _ = self.get_states(x, ctm_state, done)
        return self.critic(hidden)
    
    def get_action_and_value(self, x, ctm_state, done, action=None, track=False, action_mask=None):
        hidden, ctm_state, tracking_data = self.get_states(x, ctm_state, done, track=track)
        action_logits = self.actor(hidden)

        if self.action_type == 'bernoulli':
            action_probs = Bernoulli(logits=action_logits)

            if action is None:
                action = action_probs.sample()

            value = self.critic(hidden)

            # Sum log_prob and entropy over action dimension for multi-binary actions
            logprob = action_probs.log_prob(action).sum(dim=-1)
            entropy = action_probs.entropy().sum(dim=-1)

            return action, logprob, entropy, value, ctm_state, tracking_data, action_logits, action_probs.probs
        else:
            if action_mask is not None:
                action_logits = torch.where(action_mask, action_logits, torch.tensor(-1e8, device=action_logits.device))
            action_probs = Categorical(logits=action_logits)

        if action is None:
            action = action_probs.sample()

        value = self.critic(hidden)

        return action, action_probs.log_prob(action), action_probs.entropy(), value, ctm_state, tracking_data, action_logits, action_probs.probs