import torch
import time
from torch import optim
import os
import torch.nn as nn
import numpy as np

if os.path.basename(os.getcwd()) == 'exercises':
    os.chdir('../')

from ctm.ctm_agent import CTMAgent
from ctm.ctm_rl import ContinuousThoughtMachineRL
from ctm.img_coder import MinesweeperConvEncoder
from environments.minesweeper.minesweeper import MinesweeperEnv


def train():
    # Configure device
    device_config = -1
    if device_config != -1:
        device = f'cuda:{device_config}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model CTM on {device}')
    torch.set_default_device(device)

    # --- Training Hyperparameters ---
    num_steps = 50
    num_envs = 1
    num_minibatches = 1
    update_epochs = 1
    lr = 5e-4
    
    # Reward and Advantage Estimation
    discount_gamma = 0.99
    gae_lambda = 0.95
    
    # Loss Coefficients
    norm_adv = True
    ent_coef = 0.1
    clip_vloss = False
    clip_coef = 0.1
    vf_coef = 0.25

    # CTM & Env specific
    ctm_latent_dim = 256
    width = 6
    height = 6
    n_mines = 10

    # --- Environment Setup ---
    env = MinesweeperEnv(width, height, n_mines)
    minesweeper_enc = MinesweeperConvEncoder(ctm_latent_dim, env.state_im.shape)

    # --- Agent Initialization ---
    ctm = ContinuousThoughtMachineRL(iterations=5, 
                                   d_model=2048, 
                                   d_input=ctm_latent_dim, 
                                   n_synch_out=64, 
                                   synapse_depth=8, 
                                   memory_length=25, 
                                   deep_nlms=False,
                                   memory_hidden_dims=32, 
                                   do_layernorm_nlm=False,
                                   backbone_type='minesweeper-backbone',
                                   )

    agent = CTMAgent(ctm=ctm, continuous_state_trace=True, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # --- Training Loop ---
    total_time_steps = 5000
    num_updates = total_time_steps // (num_steps * num_envs)
    
    batch_size = num_envs * num_steps

    # Tracking variables
    global_step = 0
    start_time = time.time()
    
    # Initialize environment and agent state
    env.reset()
    state_tensor = torch.from_numpy(env.state_im.T).float().unsqueeze(0)
    next_obs = minesweeper_enc.forward(state_tensor).to(device)
    next_done = torch.zeros(num_envs).to(device)
    next_state = agent.get_initial_state(num_envs)
    
    total_wins = 0

    # Get a sample action to define shapes
    with torch.no_grad():
        sample_action, _, _, _, _, _, _, _ = agent.get_action_and_value(next_obs, next_state, next_done)

    for update in range(1, num_updates + 1):
        # --- Data Collection (Rollout) ---
        obs = torch.zeros((num_steps, num_envs, ctm_latent_dim)).to(device)
        actions = torch.zeros((num_steps, num_envs) + sample_action.shape).to(device)
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)

        initial_state = (next_state[0].clone(), next_state[1].clone())

        print("Start playing")
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state, _, _, _ = agent.get_action_and_value(next_obs, next_state, next_done)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Execute action in the environment
            # Squeeze action if it has an extra dimension
            action_to_step = action.squeeze().cpu().numpy()
            next_obs_env, reward, done = env.step(action_to_step)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # Reset if done
            if done:
                print(f"Game ended at step {step}. Reward: {reward}")
                if (reward > 0.5):
                    total_wins += 1
                env.reset()
                next_obs_env = env.state_im
                # Reset agent's memory
                next_state = agent.get_initial_state(num_envs)

            # Encode next state
            state_tensor = torch.from_numpy(next_obs_env.T).float().unsqueeze(0)
            next_obs = minesweeper_enc.forward(state_tensor).to(device)
            next_done = torch.Tensor([done]).to(device)

        print("Start learning")
        # --- Learning ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + discount_gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + discount_gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1, ctm_latent_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + sample_action.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        assert num_envs % num_minibatches == 0
        envsperbatch = num_envs // num_minibatches
        envinds = np.arange(num_envs)
        flatinds = np.arange(batch_size).reshape(num_steps, num_envs)
        
        for epoch in range(update_epochs):
            for start in range(0, num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                selected_hidden_state = (initial_state[0][:, mbenvinds, :], initial_state[1][:, mbenvinds, :])

                _, newlogprob, entropy, newvalue, _, _, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    selected_hidden_state,
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, Step {global_step}, Loss: {loss.item():.4f}, SPS: {sps}")

if __name__ == '__main__':
    train()
