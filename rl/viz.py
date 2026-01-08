
import torch
from matplotlib import pyplot as plt
import numpy as np

from rl.gcsl import GCSL

def evaluate_agent_goal_attainment(agent, env, num_runs_per_goal=100, episode_length=10):
    all_trajectories = {}

    device = agent.device
    obs_dim = env.obs_dim
    def generate_goals_and_descriptions(obs_dim, descriptions):
        """
        Generates the list of goal tensors and the corresponding plot titles.
        """
        goal_tensors = []
        goal_descriptions = []

        formatted_descriptions = [desc.replace("->", "\n->").replace("→", "\n→") for desc in descriptions]

        for i in range(obs_dim):
            goal_obs = torch.zeros((obs_dim), device=device  )
            goal_obs[i] = 1.0
            goal_tensors.append(goal_obs)
            goal_descriptions.append(formatted_descriptions[i])

        # 2. All-Zero Goal
        goal_obs_zero = torch.zeros((obs_dim), device=device)
        goal_tensors.append(goal_obs_zero)
        goal_descriptions.append("No Activity Goal: All Obs=0")
        
        # 3. All-One Goal
        goal_obs_one = torch.ones((obs_dim), device=device)
        goal_tensors.append(goal_obs_one)
        goal_descriptions.append("All Activity Goal: All Obs=1")
        return goal_tensors, goal_descriptions


    descriptions = [rule["description"] for rule in env.list_rules()]
    GOAL_TENSORS, GOAL_DESCRIPTIONS = generate_goals_and_descriptions(obs_dim, descriptions)
    NUM_GOALS = len(GOAL_TENSORS) 


    print(f"Starting experiment with {NUM_GOALS} goals. Runs per goal: {num_runs_per_goal}")

    # Iterate over the combined list of goal tensors
    for plot_index, goal_obs in enumerate(GOAL_TENSORS):
        # Determine the goal type for print statement (optional)
        if plot_index < obs_dim:
            goal_type = f"Obs[{plot_index}]=1"
            active_obs_indices = [plot_index]
        elif plot_index == obs_dim:
            goal_type = "All Zeros"
            active_obs_indices = []
        else: 
            goal_type = "All Ones"
            active_obs_indices = list(range(obs_dim))

        #print(f"  Running goal {plot_index + 1}/{NUM_GOALS}: {goal_type}")

        all_trajectories[plot_index] = (active_obs_indices, []) 
        goal_trajectories_list = all_trajectories[plot_index][1] 
        

        for run in range(num_runs_per_goal):
            obs_curr = env.reset()
            trajectory = []
            
            for step in range(episode_length):
                with torch.no_grad():
                    horizon = None
                    if isinstance(agent, GCSL) and agent.use_horizon:
                            horizon = torch.tensor([[episode_length - step]], device=device) / episode_length
                        

                    action = agent.policy.get_action(obs_curr.reshape(1, -1), goal_obs.reshape(1,-1), horizon=horizon)

                next_obs, _ = env.step(lambda o: action)
                obs_curr = next_obs
                trajectory.append(obs_curr)
                
            trajectory_s = torch.stack(trajectory) 
            goal_trajectories_list.append(trajectory_s.reshape(episode_length, -1))

   # print("Experiment complete.")

    
    # Visualization

    N_ROWS = 3
    N_COLS = 4 

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(3.5 * N_COLS, 4.5 * N_ROWS), squeeze=False)
    axes_flat = axes.flatten()

    for plot_index in range(NUM_GOALS):

        ax = axes_flat[plot_index]

        active_obs_indices, trajectories_for_goal = all_trajectories[plot_index]
        stacked_trajectories = torch.stack(trajectories_for_goal) # (num_runs_per_goal, episode_length, obs_dim)
        
        last_observation = stacked_trajectories[:, -1, :]
        target_observation = GOAL_TENSORS[plot_index].reshape(1, -1).repeat((last_observation.shape[0], 1))
        bce = torch.nn.BCELoss()(last_observation, target_observation).item()
        
        mean_trajectory = stacked_trajectories.mean(dim=0).cpu().numpy()
        plot_data = mean_trajectory.T # (obs_dim, episode_length)
        ax.imshow(plot_data, aspect='auto', interpolation='none', cmap='viridis')
        
        # --- MARK THE ACTIVE LINES ---
        if active_obs_indices:
            ax.hlines(
                active_obs_indices, # List of indices to highlight
                xmin=-0.5,
                xmax=episode_length - 0.5,
                colors='red', 
                linestyles='--', 
                linewidth=1.5,
                zorder=3 
            )

        ax.set_xlabel("Time")
        if plot_index % N_COLS == 0:
            ax.set_ylabel("Obs. Dim")
            ax.tick_params(axis='y', labelleft=True)
            ax.set_yticks(np.arange(obs_dim))
        else:
            ax.tick_params(axis='y', labelleft=False) 
            
        goal_title = GOAL_DESCRIPTIONS[plot_index] + f"\nBCE: {bce:.3f}"
        ax.set_title(
            goal_title,
            fontsize=8, 
            wrap=True,
            linespacing=1.2, 
            pad=4.0         
        )

    plt.suptitle(f"Mean Trajectories (n={num_runs_per_goal} runs) for {NUM_GOALS} Goals", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()

def plot_metrics(logger, metrics=None):
    if metrics is None:
        metrics = logger.get_metric_list()
    # 1. Create the final plot using the collected rolling average data
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), sharex=True)

    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):

        ax.plot(logger.all_logged_episodes, logger.all_avg_values[metric], label=f'{metric} (100-step Avg.)', linewidth=2)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)

        axes[-1].set_xlabel("Episode")
        fig.suptitle(f"Training Metrics (100-step average)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()





# Example usage (assuming LoggerA and LoggerB are defined)
# plot_combined_metrics_sns(LoggerA, LoggerB)