"""
    Implementation for Stable Contrastive RL based on https://github.com/chongyi-zheng/stable_contrastive_rl
    
    @misc{zheng2023stabilizing,
      title={Stabilizing Contrastive RL: Techniques for Offline Goal Reaching}, 
      author={Chongyi Zheng and Benjamin Eysenbach and Homer Walke and Patrick Yin and Kuan Fang and Ruslan Salakhutdinov and Sergey Levine},
      year={2023},
      eprint={2306.03346},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

"""
import torch
import torch.nn as nn

    

class Mlp(nn.Module):
    def __init__(self, hidden_dims, repr_shape, input_shape):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_shape, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dims[-1], repr_shape))
        self.net = nn.Sequential(*layers)
    
    def forward(self, inp):
        return self.net(inp)


class SAEncoder(nn.Module):
    def __init__(self, hidden_sizes, representation_dim, obs_dim, action_dim, obs_f = lambda x:x, a_f = lambda x:x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = Mlp(hidden_sizes, representation_dim, obs_dim+action_dim)
        self.obs_f = obs_f
        self.a_f = a_f
        
    def forward(self, state, action):
        return self.enc(torch.cat([self.obs_f(state), self.a_f(action)], dim=-1))
    
class GEncoder(nn.Module):
    def __init__(self, hidden_sizes, representation_dim, goal_dim, g_f = lambda x: x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_f = g_f
        self.enc = Mlp(hidden_sizes, representation_dim, goal_dim)
    def forward(self, goal):
        return self.enc(self.g_f(goal))


class ContrastiveQf(nn.Module):
    def __init__(self, hidden_sizes, representation_dim, action_dim, goal_dim, obs_dim, sa_encoder=None, g_encoder=None):
        
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._goal_dim = goal_dim
        self._representation_dim = representation_dim        
        if sa_encoder is None:
            self._sa_encoder = SAEncoder(hidden_sizes, representation_dim, obs_dim, action_dim)
        else:
            self._sa_encoder = sa_encoder
        if g_encoder is None:
            self._g_encoder =  GEncoder(hidden_sizes, representation_dim, goal_dim)
        else:
            self._g_encoder = g_encoder
        
    def _compute_representation(self, state, action, goal):
        sa_repr = self._sa_encoder(state, action)
        g_repr = self._g_encoder(goal)
        return sa_repr, g_repr

    def forward(self, state, action, goal):
        sa_repr, g_repr = self._compute_representation(state, action, goal)
        outer = torch.bmm(sa_repr.unsqueeze(0), g_repr.permute(1, 0).unsqueeze(0))[0]
        return outer

class StableContrastiveRL:

    def __init__(
            self,
            device,
            policy,
            qf,
            lr=3e-4,
            optimizer_class=torch.optim.Adam,
            use_adaptive_entropy_reg=True,
            adaptive_target_entropy=1.0,
            bc_coef=0.05,
            batch_size=2048,
            trajectory_length=50,
            relabel_steps=1,
            use_kl_reg=False,
            kl_target=0.0,
            initial_log_alpha=0.0,
            goal_rep=None,
            rsample=True,
            
            *args,
            **kwargs,
            
    ):
        super().__init__()
        self.policy = policy
        self.qf = qf
        self.relabel_steps = relabel_steps

        self.use_adaptive_entropy_reg = use_adaptive_entropy_reg
        self.adaptive_target_entropy = adaptive_target_entropy
        self.bc_coef = bc_coef
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length

        self.use_kl_reg = use_kl_reg
        self.kl_target = kl_target

        self.goal_rep = goal_rep
        self.rsample = rsample

        if self.use_adaptive_entropy_reg:
            self.adaptive_target_entropy = adaptive_target_entropy
            self.log_alpha = torch.full((1,), initial_log_alpha, requires_grad=True, device=device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=lr,
            )


        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=lr,
        )
        self.device = device

    def get_action(self, state, goal):
        return self.policy.get_action(state, goal)

    def update(self, buffer, step=None):
        return_steps = torch.randint(1, self.relabel_steps+1,())
        with torch.no_grad():
            dist = torch.distributions.geometric.Geometric(probs=torch.tensor(0.1))
            sample_traj_len = (dist.sample().clamp(0, self.trajectory_length)+1).int().item()  + (return_steps-1)
            transitions = buffer.sample(self.batch_size, sample_traj_len, to_device=self.device)
            
            if transitions is None:
                return {}
            

            future_transition = {}
            for key, val in transitions.items():
                future_transition[key] = val[:, 0:1]
            future_transition["future_goal"] = transitions["observation"][:,-return_steps:]
            transitions = future_transition

        loss_dict = {}

        action = transitions['action'][:,-1]
        obs = transitions["observation"][:, -1]
        goal = transitions['future_goal'][:,-1]

        batch_size = obs.shape[0]

        I = torch.eye(batch_size, device=self.device)
        logits = self.qf(obs, action, goal)


            
        # compute classifier accuracies
        with torch.no_grad():
            correct = (torch.argmax(logits, dim=-1) == torch.argmax(I, dim=-1))
            logits_pos = torch.sum(logits * I) / torch.sum(I)
            logits_neg = torch.sum(logits * (1 - I)) / torch.sum(1 - I)
            q_pos, q_neg = torch.sum(torch.sigmoid(logits) * I) / torch.sum(I), \
                        torch.sum(torch.sigmoid(logits) * (1 - I)) / torch.sum(1 - I)
            q_pos_ratio, q_neg_ratio = q_pos / (1 - q_pos), q_neg / (1 - q_neg)
            binary_accuracy = torch.mean(((logits > 0) == I).float())
            categorical_accuracy = torch.mean(correct.float())

            loss_dict["logits_pos"] = logits_pos
            loss_dict["logits_neg"] = logits_neg
            loss_dict["q_pos_ratio"] = q_pos_ratio
            loss_dict["q_neg_ratio"] = q_neg_ratio
            loss_dict["bin_acc"] = binary_accuracy
            loss_dict["cat_acc"] = categorical_accuracy
            
        # decrease the weight of negative term to 1 / (B - 1)
        qf_loss_weights = torch.ones((batch_size, batch_size), device=self.device) / (batch_size - 1)
        qf_loss_weights[torch.arange(batch_size, device=self.device), torch.arange(batch_size, device=self.device)] = 1


        qf_loss = self.qf_criterion(logits, I)
        qf_loss *= qf_loss_weights
        qf_loss = torch.mean(qf_loss)
        
        loss_dict["qf_loss"] = qf_loss.detach()


        """
        Policy and Alpha Loss
        """
        goal_rand = goal[torch.randperm(batch_size)].detach() # random goal as future goal
        
        dist = self.policy.get_dist(obs, goal_rand)
        if self.rsample:
            sampled_action = dist.rsample()
        else:
            sampled_action = dist.sample()
            
        log_prob = dist.log_prob(sampled_action)
        
        if self.use_adaptive_entropy_reg:   
            alpha_loss = -(self.log_alpha.exp() * (
                log_prob + self.adaptive_target_entropy).detach()).mean()
            alpha = self.log_alpha.exp().detach()
            loss_dict["alpha_loss"] = alpha_loss.detach()
            loss_dict["alpha"] = alpha.reshape((-1,)).detach()
            
        else:
            alpha_loss = torch.zeros((1,), device=self.device)
            alpha = 0.0 # fixed alpha

        loss_dict["actor_logp"] = log_prob.mean().detach()
        loss_dict["actor_ent"] = dist.entropy().mean().detach()
        # Actor loss: use random goals to optimize 

        q_action = self.qf(obs, sampled_action, goal_rand)
        actor_q_loss = alpha * log_prob - torch.diag(q_action)
        

        if self.use_kl_reg:   
            prior = self.policy.get_prior(dist)
            kl_div = torch.distributions.kl_divergence(dist, prior)
            kl_target = self.kl_target * torch.ones_like(kl_div)
            entropy_loss = torch.maximum(kl_div, kl_target)
            loss_dict["actor_kl"] = kl_div.mean().detach()

            actor_q_loss += entropy_loss.mean()
            
        assert 0.0 <= self.bc_coef <= 1.0
        orig_action = action

        train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()

        orig_dist = self.policy.get_dist(obs, goal.detach())
        gcbc_loss = -train_mask * orig_dist.log_prob(orig_action)
        
        actor_loss = self.bc_coef * gcbc_loss + (1 - self.bc_coef) * actor_q_loss
        actor_loss = torch.mean(actor_loss) 

        loss_dict["gcbc_loss"] = gcbc_loss.mean().detach()
        loss_dict["actor_q_loss"] = actor_q_loss.mean().detach()
        """
        Optimization.
        """
        if self.use_adaptive_entropy_reg:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return loss_dict
