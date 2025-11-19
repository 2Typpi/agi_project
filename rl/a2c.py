
import copy
import torch



class EMANetwork(torch.nn.Module):
    """
    Keeps a copy of a network with exponential moving average weights. Call update_target_network to update the weights with the given tau.
    Use forward to access the original network, use target_forward to access the EMA network.
    """

    def __init__(self, base_network, tau=0.02):
        super(EMANetwork, self).__init__()
        self.tau = tau
        self.network = base_network
        self.target_network = copy.deepcopy(base_network)

    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, *args):
        return self.network(*args)

    def target_forward(self, *args):
        return self.target_network(*args)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.network, name)
        
class Moments(torch.nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1.0,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        gathered_x = x.float().detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()

def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


class Critic(torch.nn.Module):
    def __init__(self, obs_shape, goal_shape):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape + goal_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
    
    def forward(self, obs, goal):
        out = self.net(torch.cat([obs, goal], dim=-1))
        return torch.distributions.Normal(loc=out[:, :1], scale=torch.nn.functional.softplus(out[:, 1:]) + 1e-6)
    
class AdvantageActorCritic:
    def __init__(
        self, device, policy, critic, trajectory_length, batch_size=256, adaptive_entropy_coefficient=False, kl_entropy=True, target_entropy=0.0, critic_lr=1e-3, actor_lr=1e-3, lambda_factor=0.95, discount_factor=0.95, use_moments=False, use_advantage=False, differentiable_policy=False, kl_target=1.0, kl_alpha=1.0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.policy = policy
        self.critic = critic
        self.lambda_factor = lambda_factor
        self.discount_factor = discount_factor
        self.use_moments = use_moments
        self.use_advantage = use_advantage
        self.differentiable_policy = differentiable_policy
        self.device = device
        self.moments = Moments(decay = 0.99, max_ = 1.0,
            percentile_low = 0.05, percentile_high = 0.95,)
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.opt = torch.optim.Adam(params=[
            {"params" : self.policy.parameters(), "lr" : actor_lr},
            {"params" : self.critic.parameters(), "lr" : critic_lr} 
        ], lr=1e-3)
        
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        # SAC based entropy regularization
        self.adaptive_entropy_coefficient = adaptive_entropy_coefficient
        self.kl_entropy = kl_entropy
        if self.adaptive_entropy_coefficient:
      
            self.target_entropy = target_entropy
                
            self.log_alpha = torch.full((1,), 0.0, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=3e-4,
            )
            

    def get_action(self, obs, goal):
        return self.policy.get_action(obs, goal)
    
    def compute_lambda_values(
        self, rewards, values, continues, horizon_length, device, lambda_, discount=None, abc=True, add_last_reward=1,
    ):
        """
        rewards : (batch_size, time_step, hidden_size)
        values : (batch_size, time_step, hidden_size)
        continue flag will be added
        """

        if continues is None:
            continues = discount * torch.ones_like(values)

        if abc:
            last_reward = rewards[:, -1]
            rewards = rewards[:, :-1]
            
        continues = continues[:, :-1]
        next_values = values[:, 1:]
        last = next_values[:, -1] + add_last_reward*last_reward
        inputs = rewards + continues * next_values * (1 - lambda_)

        outputs = []

        for index in reversed(range(horizon_length - 1)):
            last = inputs[:, index] + continues[:, index] * lambda_ * last
            outputs.append(last)
        returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
        return returns
    
    def first_one_indicator(self, tensor):
        first_occurrence = (tensor == 1).cumsum(dim=1).eq(1) & (tensor == 1)
        result = first_occurrence.float()
        return result

    def update(self, buffer, step):
        
        
        loss_dict = {}
        total_loss = torch.zeros((1,), device=self.device)
        unfreeze(self.policy)

        transitions = buffer.sample(
            self.batch_size, self.trajectory_length, to_device=self.device, keys=None
        )

        if transitions is None:
            return loss_dict

        freeze(self.critic)

        states = transitions["observation"]

        
        B, T = states.shape[0], states.shape[1]
        if T <= 1:
            return {}
        expanded_goals = transitions["desired_goal"].as_subclass(torch.Tensor).detach().reshape((B*T, -1))
        predicted_rewards = transitions["reward"].reshape((B,T))


        # Reward is binary (goal reached / not reached)
        first_one = self.first_one_indicator(predicted_rewards)
        reward_mask_ = ~first_one.cumsum(dim=1).bool() | first_one.bool()
        predicted_rewards = first_one


        if isinstance(self.critic, EMANetwork):
            values = self.critic.target_forward(
                states.reshape((B * T, *states.shape[2:])), expanded_goals
            ).mean.reshape((B, T))
        else:
            values = self.critic(
                states.reshape((B * T, *states.shape[2:])), expanded_goals
            ).mean.reshape((B, T))

        continues = self.discount_factor * torch.ones_like(values)

        lambda_values = self.compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            T,
            self.device,
            self.lambda_factor,
        )

        loss_dict["lambda_values"] = lambda_values.mean()

        
        # Train actor
        dist = self.policy.get_dist(
            states.reshape((B * T, *states.shape[2:])), expanded_goals
        )
        if self.use_moments:
            offset, invscale = self.moments(lambda_values)
            
            baseline = values[:, :-1]
            normed_lambda_values = (lambda_values - offset) / invscale
            normed_baseline = (baseline - offset) / invscale
            advantage = normed_lambda_values - normed_baseline

        else:
            baseline = values[:, 1:]
            advantage = lambda_values - baseline
            normed_lambda_values = lambda_values
            
        if self.use_advantage:
            advantage = advantage
        else:
            advantage = normed_lambda_values
            
        if self.differentiable_policy:
            actor_loss = -torch.mean(advantage * reward_mask_[:, :-1])
        else: 
            actions = transitions["action"].detach()
            actor_loss = -(dist.log_prob(actions.reshape((B*T,-1))).reshape((B,T,-1))[:, :-1].mean(dim=-1).reshape((-1,)) * advantage.reshape((-1))).mean()
        
        if self.adaptive_entropy_coefficient:

            sampled_action = dist.sample()
            log_prob = dist.log_prob(sampled_action)
            alpha_loss = -(self.log_alpha.exp() * (
                log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            entropy_reg = self.log_alpha.exp().detach() * log_prob.mean()

            loss_dict["actor_reg"] = entropy_reg
            loss_dict["actor_alpha"] = self.log_alpha.exp()
            loss_dict["actor_alpha_loss"] = alpha_loss
            loss_dict["actor_logp"] = log_prob.mean()

            total_loss += actor_loss + entropy_reg
            

        if self.kl_entropy:
            prior = self.policy.get_prior(dist)
            kl_div = torch.distributions.kl_divergence(dist, prior)
            kl_target = self.kl_target * torch.ones_like(kl_div)
            entropy_loss = torch.maximum(kl_div, kl_target).mean()
            loss_dict["actor_kl"] = kl_div.mean()
            total_loss += actor_loss + self.kl_alpha * entropy_loss

        entropy = dist.entropy().mean()
        loss_dict["actor_ent"] = entropy

        loss_dict["actor_loss"] = actor_loss

        # Train critic

        unfreeze(self.critic)
        freeze(self.policy)

        value_dist = self.critic(
            states[:, :-1].reshape((-1, *states.shape[2:])).detach(),
            expanded_goals.reshape((B, T, -1))[:, :-1]
            .reshape((-1, expanded_goals.shape[-1]))
            .detach(),
        )
        value_logprob = value_dist.log_prob(
            lambda_values.detach().reshape(((T - 1) * B, -1))
        ).reshape(
            B, T - 1
        )  

        value_loss = -torch.mean(value_logprob * reward_mask_[:, :-1])
        total_loss += value_loss 

        loss_dict["sample_success_rate"] = predicted_rewards.sum(dim=1).mean().detach()
        loss_dict["value_loss"] = value_loss.detach()
        loss_dict["value_mean"] = value_dist.mean.mean().detach()
        loss_dict["value_std"] = value_dist.mean.reshape((-1,)).std().detach()

        
        unfreeze(self.policy)
        unfreeze(self.critic)
            
        # Optimize
        if total_loss.requires_grad:
            self.opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                10,
                norm_type=2,
            )

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                10,
                norm_type=2,
            )

            self.opt.step()
            
        return loss_dict