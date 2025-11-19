import torch

class GCSL:

    base_config = {
        "batch_size": 256,
        "entropy_method": "kl_prior", # "kl_prior", "max_ent", "adaptive_entropy", "none"
        "kl_target": 1.0,
        "kl_scale": 1.0,
        "entropy_reg": 0.0,
        "adaptive_target_entropy": 0.0,
        "rsample": True,
    }
    def __init__(self, device, policy, goal_rep, config=base_config, use_horizon=False, trajectory_length=10):
        super().__init__()
        self.device = device
        self.policy = policy
        self.config = config
        self.goal_rep = goal_rep
        self.opt = torch.optim.Adam(params=self.policy.parameters(), lr=1e-3)
        self.batch_size = self.config["batch_size"]
        self.trajectory_length = trajectory_length
        self.use_horizon = use_horizon
        if self.config["entropy_method"] == "adaptive_entropy":
            self.adaptive_target_entropy = self.config["adaptive_target_entropy"]
            initial_log_alpha = 0.0
            self.log_alpha = torch.full((1,), initial_log_alpha, requires_grad=True, device=device)
            #self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=3e-4,
            )
            self.rsample = self.config["rsample"]



    def get_action(self, obs, goal):
        return self.policy.get_action(obs, goal)


    def update(self, buffer, step):
        loss_dict = {}
        with torch.no_grad():
            dist = torch.distributions.geometric.Geometric(probs=torch.tensor(0.1))
            sample_traj_len = (dist.sample().clamp(0, self.trajectory_length)+1).int().item()
            transitions = buffer.sample(
                self.batch_size,
                sample_traj_len,
                to_device=self.device,
                keys=None
            )
            if transitions is None:
                return {}
            relabeled_goal = transitions["observation"][:, -1].unsqueeze(1).repeat((1, sample_traj_len, 1)).reshape((-1, transitions["observation"].shape[-1]))
            observations = transitions["observation"].reshape((-1, transitions["observation"].shape[-1]))
            actions = transitions["action"].reshape((-1, transitions["action"].shape[-1]))
            if self.use_horizon:
                horizon = torch.arange(sample_traj_len-1, -1, -1, device=self.device).unsqueeze(0).repeat((self.batch_size, 1)).reshape((-1, 1)) / self.trajectory_length
                # horizon = 1 if furthest distance to goal, 0 if at goal

        if self.use_horizon:
            action_dist = self.policy.get_dist(observations, relabeled_goal, horizon)
        else:
            action_dist = self.policy.get_dist(observations, relabeled_goal)

        log_p = action_dist.log_prob(actions)#.detach()
        
        actor_loss = -log_p.mean()
        loss_dict["actor_ent"] = action_dist.entropy().mean().detach()
        if self.config["entropy_method"] == "kl_prior":
            prior = self.policy.get_prior(action_dist)
            kl_div = torch.distributions.kl_divergence(action_dist, prior)
            kl_target = self.config["kl_target"] * torch.ones_like(kl_div)
            entropy_loss = self.config["kl_scale"] * torch.maximum(kl_div, kl_target).mean()
            loss_dict["kl"] = kl_div.mean()
            
        elif self.config["entropy_method"] == "max_ent":
            actor_entropy = action_dist.entropy().mean()
            entropy_loss = -self.config["entropy_reg"] * actor_entropy
        elif self.config["entropy_method"] == "adaptive_entropy":
            if self.rsample:
                sampled_action = action_dist.rsample()#.clamp(-1.0+1e-6, 1.0-1e-6)
            else:
                sampled_action = action_dist.sample()#.clamp(-1.0+1e-6, 1.0-1e-6)

            #sampled_action = torch.clamp(sampled_action, -1.0+1e-6, 1.0-1e-6)
            log_prob = action_dist.log_prob(sampled_action)#.mean(dim=-1) # differentiate through actions?
            #log_prob = torch.clamp(log_prob, -20, 0)
            
            alpha_loss = -(self.log_alpha.exp() * (log_prob + self.adaptive_target_entropy).detach()).mean()
            alpha = self.log_alpha.exp().detach()
            loss_dict["alpha_loss"] = alpha_loss.detach()
            loss_dict["alpha"] = alpha.detach()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            entropy_loss = (alpha * log_prob).mean()
            
        else:
            entropy_loss = None
        
        total_loss = actor_loss + entropy_loss if entropy_loss is not None else actor_loss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return loss_dict | {"actor_loss": actor_loss}#, "actor_entropy": actor_entropy}
