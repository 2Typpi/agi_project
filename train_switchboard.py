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
from environments.switchboard.switchboard import Switchboard


def save_model(agent, optimizer, global_step, update, episode_returns,
               episode_slots_activated, slot_activation_history, update_logs, save_path):
    """Save checkpoint (no external encoder for switchboard)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'update': update,
        'episode_returns': episode_returns,
        'episode_slots_activated': episode_slots_activated,
        'slot_activation_history': slot_activation_history,
        'update_logs': update_logs,
    }, save_path)


def load_model(agent, optimizer, checkpoint_path, device):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    update = checkpoint.get('update', 0)
    episode_returns = checkpoint.get('episode_returns', [])
    episode_slots_activated = checkpoint.get('episode_slots_activated', [])
    slot_activation_history = checkpoint.get('slot_activation_history', [])
    update_logs = checkpoint.get('update_logs', [])

    print(f"Loaded checkpoint from {checkpoint_path} at update {update}, step {global_step}")
    return global_step, update, episode_returns, episode_slots_activated, slot_activation_history, update_logs


def compute_reward(prev_obs, next_obs):
    """
    Reward function: +1 for each newly activated observation slot.
    This encourages the agent to discover rules and activate slots.

    Args:
        prev_obs: Previous observation tensor (num_envs, obs_dim)
        next_obs: Next observation tensor (num_envs, obs_dim)

    Returns:
        reward: Reward tensor (num_envs,)
    """
    newly_activated = ((next_obs > 0.5) & (prev_obs <= 0.5)).float()
    return newly_activated.sum(dim=-1)


def plot_results(episode_returns, episode_slots_activated, update_logs, save_dir="./plots"):
    """Generate training curves for switchboard training."""
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("CTM PPO — Switchboard Training", fontsize=14, fontweight='bold')

    if episode_returns:
        ep_steps = np.array([r[0] for r in episode_returns])
        ep_rets = np.array([r[1] for r in episode_returns], dtype=float)
        ep_slots = np.array([s[1] for s in episode_slots_activated], dtype=float)

        mean_ret, std_ret = rolling(ep_rets, EP_WINDOW)
        mean_slots, std_slots = rolling(ep_slots, EP_WINDOW)

        # Average Reward
        ax = axes[0, 0]
        ax.scatter(ep_steps, ep_rets, alpha=0.15, s=4, color='steelblue', linewidths=0)
        ax.plot(ep_steps, mean_ret, color='steelblue', linewidth=1.8, label='Mean reward')
        ax.fill_between(ep_steps, mean_ret - std_ret, mean_ret + std_ret,
                        alpha=0.2, color='steelblue', label='±1 std')
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Episode Reward (rolling {EP_WINDOW} episodes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Unique Slots Activated
        ax = axes[0, 1]
        ax.scatter(ep_steps, ep_slots, alpha=0.15, s=4, color='darkorange', linewidths=0)
        ax.plot(ep_steps, mean_slots, color='darkorange', linewidth=1.8, label='Mean slots')
        ax.fill_between(ep_steps, mean_slots - std_slots, mean_slots + std_slots,
                        alpha=0.2, color='darkorange', label='±1 std')
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Unique Slots Activated")
        ax.set_title(f"Unique Slots Activated (rolling {EP_WINDOW} episodes)")
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
    else:
        for ax in axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]:
            ax.text(0.5, 0.5, "No update data", ha='center', va='center',
                    transform=ax.transAxes)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "training_curves.png")
    pdf_path = os.path.join(save_dir, "training_curves.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {png_path} and {pdf_path}")


def plot_slot_discovery(slot_activation_history, save_dir="./plots"):
    """
    Generate per-slot activation visualizations showing which rules were discovered.

    Creates three plots:
    1. Slot Activation Timeline (scatter): When each slot was active
    2. Slot Frequency Heatmap: Activation rate per training window
    3. Cumulative Discovery: Number of unique slots discovered over time
    """
    if not slot_activation_history:
        print("No slot activation history to plot")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Rule labels for each slot
    slot_labels = [
        "0: Direct btn 0",
        "1: Direct btn 1",
        "2: 3-step delay",
        "3: 5-step delay",
        "4: 8-step delay",
        "5: AND(5+6)",
        "6: Seq(7→8→9)",
        "7: Hold btn 9",
        "8: Unused",
        "9: Unused"
    ]

    # Extract data
    all_steps = np.array([x[0] for x in slot_activation_history])
    all_slot_sets = [set(x[1]) for x in slot_activation_history]

    # Build activation matrix: (num_episodes, 10 slots)
    num_episodes = len(all_slot_sets)
    activation_matrix = np.zeros((num_episodes, 10))
    for ep_idx, slots in enumerate(all_slot_sets):
        for slot in slots:
            activation_matrix[ep_idx, slot] = 1

    # Setup plotting style
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('bmh')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CTM PPO — Slot Discovery Analysis", fontsize=14, fontweight='bold')

    # Plot 1: Slot Activation Timeline (Scatter)
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for slot_idx in range(10):
        # Find episodes where this slot was active
        active_episodes = np.where(activation_matrix[:, slot_idx] > 0)[0]
        if len(active_episodes) > 0:
            ax.scatter(all_steps[active_episodes],
                      np.full(len(active_episodes), slot_idx),
                      s=3, alpha=0.6, color=colors[slot_idx],
                      label=f"Slot {slot_idx}")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Slot Index")
    ax.set_title("Slot Activation Timeline")
    ax.set_ylim(-0.5, 9.5)
    ax.set_yticks(range(10))
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Slot Frequency Heatmap
    ax = axes[1]

    # Bin episodes into training windows
    num_windows = 10
    max_step = all_steps[-1] if len(all_steps) > 0 else 100000
    window_edges = np.linspace(0, max_step, num_windows + 1)
    window_centers = (window_edges[:-1] + window_edges[1:]) / 2

    # Compute activation frequency per window
    heatmap_data = np.zeros((10, num_windows))
    for window_idx in range(num_windows):
        window_mask = (all_steps >= window_edges[window_idx]) & (all_steps < window_edges[window_idx + 1])
        if window_mask.sum() > 0:
            heatmap_data[:, window_idx] = activation_matrix[window_mask].mean(axis=0) * 100

    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_xlabel("Training Progress")
    ax.set_ylabel("Slot (Rule)")
    ax.set_title("Slot Activation Frequency (%)")
    ax.set_yticks(range(10))
    ax.set_yticklabels(slot_labels, fontsize=8)
    ax.set_xticks(range(num_windows))
    ax.set_xticklabels([f"{int(window_centers[i]/1000)}k" for i in range(num_windows)],
                       rotation=45, ha='right', fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation %', rotation=270, labelpad=15)

    # Plot 3: Cumulative Discovery
    ax = axes[2]

    # Compute cumulative unique slots discovered
    cumulative_slots = np.zeros(num_episodes)
    discovered = set()
    for ep_idx, slots in enumerate(all_slot_sets):
        discovered.update(slots)
        cumulative_slots[ep_idx] = len(discovered)

    ax.plot(all_steps, cumulative_slots, color='steelblue', linewidth=2)
    ax.axhline(8, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Theoretical Max (8 rules)')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Unique Slots Discovered")
    ax.set_title("Cumulative Rule Discovery")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "slot_discovery.png")
    pdf_path = os.path.join(save_dir, "slot_discovery.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Slot discovery plots saved to {png_path} and {pdf_path}")

    # Print summary statistics
    print("\n=== Slot Discovery Report ===")
    for slot_idx in range(10):
        activation_rate = activation_matrix[:, slot_idx].mean() * 100
        status = "✓" if activation_rate > 50 else "✗"
        print(f"{slot_labels[slot_idx]}: {activation_rate:5.1f}% {status}")
    print(f"\nOverall: {int(cumulative_slots[-1])}/8 rules discovered")


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
    num_steps = 128
    num_envs = 16
    num_minibatches = 4
    update_epochs = 4
    lr = 3e-4
    save_interval = 100

    # Reward and Advantage Estimation
    discount_gamma = 0.99
    gae_lambda = 0.95

    # Loss Coefficients
    norm_adv = True
    ent_coef = 0.05
    clip_vloss = True
    clip_coef = 0.2
    vf_coef = 0.5

    # Environment settings
    action_dim = 10
    obs_dim = 10
    episode_max_steps = 30

    # CTM settings
    ctm_d_model = 512
    ctm_latent_dim = obs_dim
    ctm_iterations = 8
    ctm_memory_length = 20

    # --- Environment Setup ---
    envs = [
        Switchboard(
            action_dim=action_dim,
            obs_dim=obs_dim,
            scenario='temporal_ppo',
            time_scaling=0.0,
            device=device
        )
        for _ in range(num_envs)
    ]

    # --- Agent Initialization ---
    ctm = ContinuousThoughtMachineRL(
        iterations=ctm_iterations,
        d_model=ctm_d_model,
        d_input=ctm_latent_dim,
        n_synch_out=64,
        synapse_depth=6,
        memory_length=ctm_memory_length,
        deep_nlms=False,
        memory_hidden_dims=64,
        do_layernorm_nlm=True,
        backbone_type='classic-control-backbone',
    )

    agent = CTMAgent(
        ctm=ctm,
        continuous_state_trace=True,
        device=device,
        num_actions=action_dim,
        action_type='bernoulli'
    )

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    checkpoint_path = "./models/switchboard_checkpoint.pt"
    global_step = 0
    start_update = 0
    episode_returns = []
    episode_slots_activated = []
    slot_activation_history = []
    update_logs = []

    if os.path.exists(checkpoint_path):
        global_step, start_update, episode_returns, episode_slots_activated, slot_activation_history, update_logs = \
            load_model(agent, optimizer, checkpoint_path, device)
    else:
        print("No checkpoint found, starting fresh.")

    total_time_steps = 1_000_000
    num_updates = total_time_steps // (num_steps * num_envs)

    batch_size = num_envs * num_steps

    start_time = time.time()

    # Initialize environments and agent state
    next_obs = torch.stack([env.reset() for env in envs]).to(device)
    next_done = torch.zeros(num_envs).to(device)
    next_state = agent.get_initial_state(num_envs)

    ep_reward_buf = np.zeros(num_envs)
    ep_slots_buf = [set() for _ in range(num_envs)]
    ep_step_count = np.zeros(num_envs, dtype=int)

    for update in range(start_update + 1, num_updates + 1):
        current_ent_coef = ent_coef * max(0.2, 1.0 - (update / num_updates) * 0.8)

        obs_buffer = torch.zeros((num_steps, num_envs, obs_dim)).to(device)
        actions = torch.zeros((num_steps, num_envs, action_dim)).to(device)
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)

        initial_state = (next_state[0].clone(), next_state[1].clone())

        print("Start playing")
        for step in range(0, num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones[step] = next_done

            # Store previous observation for reward computation
            prev_obs = next_obs.clone()

            with torch.no_grad():
                action, logprob, _, value, next_state, _, _, _ = agent.get_action_and_value(
                    next_obs, next_state, next_done
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute actions across all environments
            new_obs_list = []
            new_rewards = []
            new_dones = []

            for i, env in enumerate(envs):
                observations, _ = env.step(lambda _: action[i])
                next_obs_i = observations[-1]

                # Compute reward based on newly activated slots
                reward_i = compute_reward(prev_obs[i:i+1], next_obs_i.unsqueeze(0))
                ep_reward_buf[i] += reward_i.item()

                # Track unique slots activated
                active_slots = (next_obs_i > 0.5).nonzero(as_tuple=True)[0]
                for slot_idx in active_slots:
                    ep_slots_buf[i].add(slot_idx.item())

                # Manual episode termination
                ep_step_count[i] += 1
                done_i = (ep_step_count[i] >= episode_max_steps)

                if done_i:
                    episode_returns.append((global_step, ep_reward_buf[i]))
                    episode_slots_activated.append((global_step, len(ep_slots_buf[i])))
                    slot_activation_history.append((global_step, sorted(list(ep_slots_buf[i]))))
                    ep_reward_buf[i] = 0.0
                    ep_slots_buf[i] = set()
                    ep_step_count[i] = 0
                    next_obs_i = env.reset()

                new_obs_list.append(next_obs_i)
                new_rewards.append(reward_i.item())
                new_dones.append(float(done_i))

            rewards[step] = torch.tensor(new_rewards, device=device)
            next_done = torch.tensor(new_dones, device=device)
            next_obs = torch.stack(new_obs_list).to(device)

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
        b_obs = obs_buffer.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
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

                # CRITICAL: Index dimension 0 (num_envs), not d_model
                selected_hidden_state = (initial_state[0][mbenvinds], initial_state[1][mbenvinds])

                _, newlogprob, entropy, newvalue, _, _, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    selected_hidden_state,
                    b_dones[mb_inds],
                    b_actions[mb_inds],
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
                loss = pg_loss - current_ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                mb_pg_losses.append(pg_loss.item())
                mb_v_losses.append(v_loss.item())
                mb_entropies.append(entropy_loss.item())

        with torch.no_grad():
            ev = 1.0 - (b_returns - b_values).var() / (b_returns.var() + 1e-8)

        update_logs.append({
            'step': global_step,
            'pg_loss': float(np.mean(mb_pg_losses)),
            'v_loss': float(np.mean(mb_v_losses)),
            'entropy': float(np.mean(mb_entropies)),
            'explained_variance': float(ev.item()),
        })

        sps = int(global_step / (time.time() - start_time))
        avg_reward = np.mean([r[1] for r in episode_returns[-100:]]) if episode_returns else 0.0
        avg_slots = np.mean([s[1] for s in episode_slots_activated[-100:]]) if episode_slots_activated else 0.0
        print(f"Update {update}/{num_updates}, Step {global_step}, Loss: {loss.item():.4f}, "
              f"EV: {ev.item():.3f}, AvgReward: {avg_reward:.2f}, AvgSlots: {avg_slots:.2f}, "
              f"EntCoef: {current_ent_coef:.4f}, SPS: {sps}")

        # Periodic checkpoint saving
        if save_interval > 0 and update % save_interval == 0:
            print(f"Saving checkpoint at update {update}...")
            save_model(agent, optimizer, global_step, update, episode_returns,
                       episode_slots_activated, slot_activation_history, update_logs, checkpoint_path)
            plot_results(episode_returns, episode_slots_activated, update_logs)
            plot_slot_discovery(slot_activation_history)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
    print(f"Saving final checkpoint...")
    save_model(agent, optimizer, global_step, update, episode_returns,
               episode_slots_activated, slot_activation_history, update_logs, checkpoint_path)

    plot_results(episode_returns, episode_slots_activated, update_logs)
    plot_slot_discovery(slot_activation_history)


if __name__ == '__main__':
    train()
