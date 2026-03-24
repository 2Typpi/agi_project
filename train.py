import torch
import time
from torch import optim
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if os.path.basename(os.getcwd()) == 'exercises':
    os.chdir('../')

from ctm.ctm_agent import CTMAgent
from ctm.ctm_rl import ContinuousThoughtMachineRL
from ctm.img_coder import MinesweeperConvEncoder
from environments.minesweeper.minesweeper import MinesweeperEnv


def save_model(agent, minesweeper_enc, optimizer, global_step, update, total_wins,
               episode_returns, episode_wins, update_logs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': agent.state_dict(),
        'encoder_state_dict': minesweeper_enc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'update': update,
        'total_wins': total_wins,
        'episode_returns': episode_returns,
        'episode_wins': episode_wins,
        'update_logs': update_logs,
    }, save_path)


def load_model(agent, minesweeper_enc, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    minesweeper_enc.load_state_dict(checkpoint['encoder_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    update = checkpoint.get('update', 0)
    total_wins = checkpoint.get('total_wins', 0)
    episode_returns = checkpoint.get('episode_returns', [])
    episode_wins = checkpoint.get('episode_wins', [])
    update_logs = checkpoint.get('update_logs', [])

    print(f"Loaded checkpoint from {checkpoint_path} at update {update}, step {global_step}, wins {total_wins}")
    return global_step, update, total_wins, episode_returns, episode_wins, update_logs


def plot_results(episode_returns, episode_wins, update_logs, save_dir="./plots"):
    """Generate publication-quality training curves and save to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    def rolling(values, window):
        """Rolling mean and std of same length as input."""
        arr = np.array(values, dtype=float)
        mean = np.empty(len(arr))
        std = np.empty(len(arr))
        for i in range(len(arr)):
            window_data = arr[max(0, i - window + 1):i + 1]
            mean[i] = window_data.mean()
            std[i] = window_data.std()
        return mean, std

    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('bmh')

    EP_WINDOW = 100
    UPD_WINDOW = 10

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle("CTM PPO — Minesweeper Training", fontsize=14, fontweight='bold')

    if episode_returns:
        ep_steps = np.array([r[0] for r in episode_returns])
        ep_rets = np.array([r[1] for r in episode_returns], dtype=float)
        ep_wins_arr = np.array([w[1] for w in episode_wins], dtype=float)

        mean_wr, _ = rolling(ep_wins_arr, EP_WINDOW)
        mean_ret, std_ret = rolling(ep_rets, EP_WINDOW)

        # Win Rate
        ax = axes[0, 0]
        ax.plot(ep_steps, mean_wr, color='steelblue', linewidth=1.5)
        ax.fill_between(ep_steps, np.clip(mean_wr - 0.05, 0, 1),
                        np.clip(mean_wr + 0.05, 0, 1), alpha=0.2, color='steelblue')
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Win Rate (rolling {EP_WINDOW} episodes)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Episode Return
        ax = axes[0, 1]
        ax.scatter(ep_steps, ep_rets, alpha=0.15, s=4, color='darkorange', linewidths=0)
        ax.plot(ep_steps, mean_ret, color='darkorange', linewidth=1.8, label='Mean return')
        ax.fill_between(ep_steps, mean_ret - std_ret, mean_ret + std_ret,
                        alpha=0.2, color='darkorange', label='±1 std')
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"Episode Return (rolling {EP_WINDOW} episodes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "No episode data", ha='center', va='center',
                        transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, "No episode data", ha='center', va='center',
                        transform=axes[0, 1].transAxes)

    if update_logs:
        upd_steps = np.array([u['step'] for u in update_logs])
        pg_losses = np.array([u['pg_loss'] for u in update_logs])
        v_losses = np.array([u['v_loss'] for u in update_logs])
        entropies = np.array([u['entropy'] for u in update_logs])
        evs = np.array([u['explained_variance'] for u in update_logs])

        mean_ev, _ = rolling(evs, UPD_WINDOW)
        mean_pg, _ = rolling(pg_losses, UPD_WINDOW)
        mean_vl, _ = rolling(v_losses, UPD_WINDOW)
        mean_ent, _ = rolling(entropies, UPD_WINDOW)

        # Explained Variance
        ax = axes[0, 2]
        ax.plot(upd_steps, mean_ev, color='seagreen', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Value Function — Explained Variance")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

        # Policy Loss
        ax = axes[1, 0]
        ax.plot(upd_steps, mean_pg, color='crimson', linewidth=1.5)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss (PPO Clipped)")
        ax.grid(True, alpha=0.3)

        # Value Loss
        ax = axes[1, 1]
        ax.plot(upd_steps, mean_vl, color='mediumpurple', linewidth=1.5)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Value Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

        # Entropy
        ax = axes[1, 2]
        ax.plot(upd_steps, mean_ent, color='teal', linewidth=1.5)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

        # Mean Episode Reward
        if 'mean_ep_reward' in update_logs[0]:
            mean_ep_rews = np.array([u['mean_ep_reward'] for u in update_logs])
            mean_epr, _ = rolling(mean_ep_rews, UPD_WINDOW)

            ax = axes[2, 0]
            ax.plot(upd_steps, mean_epr, color='darkgoldenrod', linewidth=1.5)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Mean Episode Reward")
            ax.set_title("Mean Episode Reward (rolling 100 episodes)")
            ax.grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, "No mean reward data", ha='center', va='center',
                           transform=axes[2, 0].transAxes)
    else:
        for ax in axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]:
            ax.text(0.5, 0.5, "No update data", ha='center', va='center',
                    transform=ax.transAxes)

    # Hide unused subplots
    for ax in [axes[2, 1], axes[2, 2]]:
        ax.axis('off')

    plt.tight_layout()
    png_path = os.path.join(save_dir, "training_curves.png")
    pdf_path = os.path.join(save_dir, "training_curves.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {png_path} and {pdf_path}")


def train():
    device_config = 0
    if torch.cuda.is_available() and device_config is not None and device_config >= 0:
        device = f'cuda:{device_config}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model CTM on {device}')
    torch.set_default_device(device)

    # --- Training Hyperparameters ---
    num_steps = 128
    num_envs = 8
    num_minibatches = 4
    update_epochs = 4
    lr = 3e-4
    save_interval = 100

    # Reward and Advantage Estimation
    discount_gamma = 0.99
    gae_lambda = 0.95

    # Loss Coefficients
    norm_adv = True
    clip_vloss = True
    clip_coef = 0.2
    vf_coef = 0.5

    # CTM & Env specific
    ctm_latent_dim = 256
    width = 4
    height = 4
    n_mines = 4

    # --- Environment Setup ---
    envs = [MinesweeperEnv(width, height, n_mines) for _ in range(num_envs)]
    minesweeper_enc = MinesweeperConvEncoder(ctm_latent_dim, envs[0].state_im.shape)

    # --- Agent Initialization ---
    ctm = ContinuousThoughtMachineRL(iterations=8,
                                   d_model=1024,
                                   d_input=ctm_latent_dim,
                                   n_synch_out=64,
                                   synapse_depth=6,
                                   memory_length=20,
                                   deep_nlms=False,
                                   memory_hidden_dims=64,
                                   do_layernorm_nlm=True,
                                   backbone_type='minesweeper-backbone',
                                   )

    agent = CTMAgent(ctm=ctm, continuous_state_trace=True, device=device, num_actions=envs[0].ntiles)
    all_params = list(agent.parameters()) + list(minesweeper_enc.parameters())
    optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)

    checkpoint_path = "./models/checkpoint.pt"
    global_step = 0
    start_update = 0
    total_wins = 0
    episode_returns = []
    episode_wins = []
    update_logs = []

    if os.path.exists(checkpoint_path):
        global_step, start_update, total_wins, episode_returns, episode_wins, update_logs = \
            load_model(agent, minesweeper_enc, optimizer, checkpoint_path, device)
    else:
        print("No checkpoint found, starting fresh.")

    total_time_steps = 1_000_000
    num_updates = total_time_steps // (num_steps * num_envs)

    batch_size = num_envs * num_steps

    start_time = time.time()

    # Initialize environments and agent state
    for env in envs:
        env.reset()
    raw_next_obs = torch.stack([
        torch.from_numpy(env.state_im.T).float() for env in envs
    ]).to(device)
    with torch.no_grad():
        next_obs = minesweeper_enc(raw_next_obs)
    next_done = torch.zeros(num_envs).to(device)
    next_state = agent.get_initial_state(num_envs)

    ep_reward_buf = np.zeros(num_envs)

    for update in range(start_update + 1, num_updates + 1):
        # --- Data Collection ---
        raw_obs = torch.zeros((num_steps, num_envs, 1, height, width)).to(device)
        actions = torch.zeros((num_steps, num_envs)).to(device)
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)

        initial_state = (next_state[0].clone(), next_state[1].clone())

        print("Start playing")
        for step in range(0, num_steps):
            global_step += num_envs
            raw_obs[step] = raw_next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state, _, _, _ = agent.get_action_and_value(next_obs, next_state, next_done)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute actions across all environments
            new_raw_obs_list = []
            new_rewards = []
            new_dones = []

            for i, env in enumerate(envs):
                next_obs_env_i, reward_i, done_i = env.step(int(action[i].item()))
                ep_reward_buf[i] += reward_i

                if done_i:
                    is_win = reward_i > 0.5
                    if is_win:
                        total_wins += 1
                    episode_returns.append((global_step, ep_reward_buf[i]))
                    episode_wins.append((global_step, int(is_win)))
                    ep_reward_buf[i] = 0.0
                    env.reset()
                    next_obs_env_i = env.state_im

                new_raw_obs_list.append(torch.from_numpy(next_obs_env_i.T).float())
                new_rewards.append(reward_i)
                new_dones.append(float(done_i))

            rewards[step] = torch.tensor(new_rewards, device=device)
            next_done = torch.tensor(new_dones, device=device)

            raw_next_obs = torch.stack(new_raw_obs_list).to(device)
            with torch.no_grad():
                next_obs = minesweeper_enc(raw_next_obs)

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
        b_raw_obs = raw_obs.reshape((-1, 1, height, width))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        assert num_envs % num_minibatches == 0
        envsperbatch = num_envs // num_minibatches
        envinds = np.arange(num_envs)
        flatinds = np.arange(batch_size).reshape(num_steps, num_envs)

        mb_pg_losses, mb_v_losses, mb_entropies = [], [], []

        for epoch in range(update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                selected_hidden_state = (initial_state[0][mbenvinds], initial_state[1][mbenvinds])

                mb_obs = minesweeper_enc(b_raw_obs[mb_inds])

                _, newlogprob, entropy, newvalue, _, _, _, _ = agent.get_action_and_value(
                    mb_obs,
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
                loss = pg_loss - entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 0.5)
                optimizer.step()

                mb_pg_losses.append(pg_loss.item())
                mb_v_losses.append(v_loss.item())
                mb_entropies.append(entropy_loss.item())

        with torch.no_grad():
            ev = 1.0 - (b_returns - b_values).var() / (b_returns.var() + 1e-8)

        # Compute mean episode reward from recent 100 episodes
        mean_ep_reward = 0.0
        if len(episode_returns) > 0:
            recent_rewards = [r[1] for r in episode_returns[-100:]]
            mean_ep_reward = np.mean(recent_rewards)

        update_logs.append({
            'step': global_step,
            'pg_loss': float(np.mean(mb_pg_losses)),
            'v_loss': float(np.mean(mb_v_losses)),
            'entropy': float(np.mean(mb_entropies)),
            'explained_variance': float(ev.item()),
            'mean_ep_reward': float(mean_ep_reward),
        })

        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, Step {global_step}, Loss: {loss.item():.4f}, "
              f"EV: {ev.item():.3f}, SPS: {sps}, Wins: {total_wins}, Mean Ep Reward: {mean_ep_reward:.3f}")

        # Periodic checkpoint saving
        if save_interval > 0 and update % save_interval == 0:
            print(f"Saving checkpoint at update {update}...")
            save_model(agent, minesweeper_enc, optimizer, global_step, update, total_wins,
                       episode_returns, episode_wins, update_logs, checkpoint_path)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
    print(f"Saving final checkpoint...")
    save_model(agent, minesweeper_enc, optimizer, global_step, update, total_wins,
                episode_returns, episode_wins, update_logs, checkpoint_path)

    plot_results(episode_returns, episode_wins, update_logs)


if __name__ == '__main__':
    train()