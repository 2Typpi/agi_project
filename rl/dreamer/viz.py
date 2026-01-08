import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torch

def loggers_to_dataframe(loggers, metric, logger_labels):
    """
    Converts data from multiple logger objects for a single metric into a tidy DataFrame.
    """
    data = []
    for logger, label in zip(loggers, logger_labels):
        # Check if the metric exists in the logger before creating the DataFrame
        if metric in logger.all_avg_values:
            df_temp = pd.DataFrame({
                'Episode': logger.all_logged_episodes,
                'Value': logger.all_avg_values[metric],
                'Logger': label
            })
            data.append(df_temp)
    return pd.concat(data, ignore_index=True)

def plot_combined_metrics_sns_single_legend(
    logger1, logger2, metrics=None, titles=None,
    logger1_label="Model A", logger2_label="Model B"
):
    """
    Plot multiple metrics side-by-side using matplotlib, matching the 
    same visual style as the rollout accuracy plot:
    
    - Matplotlib line plots (no seaborn lineplot)
    - Circle markers ('o'), markersize=4
    - Legend centered above plots
    - Whitegrid style
    - Thin grid lines
    """

    # 🎨 Match rollout plot style
    sns.set_theme(style="whitegrid")

    loggers = [logger1, logger2]
    logger_labels = [logger1_label, logger2_label]

    if metrics is None:
        metrics = logger1.get_metric_list()

    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 3), sharey=True)

    if num_metrics == 1:
        axes = [axes]

    handles = []
    labels = []

    # Colors from matplotlib default cycle (same as rollout accuracy plot)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (ax, metric) in enumerate(zip(axes, metrics)):

        # Convert metric logs → simple arrays
        data_frames = []
        for logger, lbl in zip(loggers, logger_labels):
            df = loggers_to_dataframe([logger], metric, [lbl])
            data_frames.append(df)

        plot_data = pd.concat(data_frames, ignore_index=True)

        # Manual plotting using matplotlib (NOT seaborn lineplot)
        for j, lbl in enumerate(logger_labels):
            df = plot_data[plot_data["Logger"] == lbl]

            h = ax.plot(
                df["Episode"],
                df["Value"],
                label=lbl,
                #marker="o",
                #markersize=1,
                linewidth=3.0,
                color=colors[j % len(colors)],   # matches rollout plot colors
            )[0]

            if i == 0:  # capture legend only once
                handles.append(h)
                labels.append(lbl)

        # Titles and formatting
        title = metric.replace("_", " ").title() if titles is None else titles[i]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 20000)
        ax.set_xticks([0, 10000, 20000]) # make it e10
        ax.set_xticklabels(['0', '1e4', '2e4'])
        ax.grid(True, linestyle="--", alpha=0.6)

        if i == 0:
            ax.set_ylabel("Accuracy", fontsize=11)

    # === Central legend (same as rollout accuracy plot) ===
# === Central legend (below the plots, same as rollout accuracy plot) ===
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(labels)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    #plt.savefig("exercises/dreamer_data/training_prediction_accuracy_comparison.png", dpi=300, bbox_inches='tight')  
    
def save_dreamer_models(agent, logger, energy_agent, energy_logger):
    agent.logger = logger
    energy_agent.logger = energy_logger
    torch.save(agent, "exercises/dreamer_data/dreamer_agent.pt")
    torch.save(energy_agent, "exercises/dreamer_data/energy_dreamer_agent.pt")

def load_dreamer_models():
    agent = torch.load("exercises/dreamer_data/dreamer_agent.pt", weights_only=False)
    energy_agent = torch.load("exercises/dreamer_data/energy_dreamer_agent.pt", weights_only=False)
    return agent, energy_agent


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch 

def plot_comparison_sequence_simultaneous(env, agent1, agent2, device, num_steps=10, agent1_label="Agent 1", agent2_label="Agent 2", N_grounding_steps=5):
    
    labels = [agent1_label, agent2_label]
    
    sequences = {
        label: {'rec_observations': [], 'real_observations': []}
        for label in labels
    }
    
    agent1.dreamer.reset()
    agent2.dreamer.reset()
    obs = env.reset() 
    
    agent1.dreamer.state = agent1.dreamer.state.clone()
    agent2.dreamer.state = agent2.dreamer.state.clone()
    
    
    # --- Grounding Phase ---
    print(f"Grounding agents for {N_grounding_steps} steps...")

    for i in range(N_grounding_steps):
        action = torch.distributions.OneHotCategorical(logits=torch.ones((1, 4), device=device)).sample()
        real_observation = env.step(lambda o: action)[0]
        
        agent1.dreamer.state = agent1.dreamer.step(action, real_observation)
        agent2.dreamer.state = agent2.dreamer.step(action, real_observation)

        
    # --- Simultaneous Data Collection Loop ---
    print(f"Collecting and processing simultaneous trajectory over {num_steps} steps...")
    
    shared_real_observations = [] 

    for i in range(num_steps):
        action = torch.distributions.OneHotCategorical(logits=torch.ones((1, 4), device=device)).sample()
        real_observation = env.step(lambda o: action)[0]
        
        agent1.dreamer.state = agent1.dreamer.dream_step(action, agent1.dreamer.state)
        rec_observation1 = agent1.dreamer.rssm.decoder(agent1.dreamer.state.combined)
        
        agent2.dreamer.state = agent2.dreamer.dream_step(action, agent2.dreamer.state)
        rec_observation2 = agent2.dreamer.rssm.decoder(agent2.dreamer.state.combined)
        
        sequences[labels[0]]['rec_observations'].append(rec_observation1.squeeze(0).detach().cpu().numpy())
        sequences[labels[1]]['rec_observations'].append(rec_observation2.squeeze(0).detach().cpu().numpy())
        
        shared_real_observations.append(real_observation.squeeze(0).detach().cpu().numpy())


    # --- Plotting ---
    
    fig, axs = plt.subplots(3, num_steps, figsize=(num_steps * 2, 8)) 
    

    plot_rows = [
        (0, labels[0], 'Imagined'),
        (1, labels[1], 'Imagined'),
        (2, 'Real', 'Real')
    ]

    for row_idx, agent_label, observation_type in plot_rows:
        
        if observation_type == 'Imagined':
            data_list = sequences[agent_label]['rec_observations']
        else:
            data_list = shared_real_observations
            
        for i in range(num_steps):
            ax = axs[row_idx, i]
            img_data = data_list[i]
            
            img = img_data.transpose(1, 2, 0).clip(0, 1)

            # ======================================
            # INLINE ROUNDED BORDER AROUND EACH AXES
            # ======================================


            border = FancyBboxPatch(
                (0.02, 0.02),   # inset inside axes
                0.96, 0.96,     # border spans almost entire image
                boxstyle="round,pad=0.02,rounding_size=0.1",  # smooth, modern rounding
                linewidth=3,
                edgecolor="#1f77b4",   # or use agent color dynamically
                facecolor="none",
                alpha=0.9,
                transform=ax.transAxes,   # IMPORTANT: consistent look across axes!
                zorder=10
            )
            ax.add_patch(border)

            ax.imshow(img)
            ax.axis('off')

        # Row labels
        if observation_type == 'Imagined':
            axs[row_idx, 0].text(-0.15, 0.5, agent_label,
                                 va='center', ha='right',
                                 fontsize=18, fontweight='bold',
                                 rotation=90, transform=axs[row_idx, 0].transAxes)
            axs[row_idx, 0].text(-0.35, 0.5, "(Imagined)",
                                 va='center', ha='right',
                                 fontsize=18, rotation=90,
                                 transform=axs[row_idx, 0].transAxes)
        else:
            axs[row_idx, 0].text(-0.15, 0.5, "Real",
                                 va='center', ha='right',
                                 fontsize=18, fontweight='bold',
                                 rotation=90, transform=axs[row_idx, 0].transAxes)

    # ======================================
    # INLINE ARROW (YOUR EXACT STYLE)
    # ======================================
    from matplotlib.patches import FancyArrowPatch
    ACCENT = "#1f77b4"

    arrow_x = FancyArrowPatch(
        posA=(0.15, -0.04),
        posB=(0.95, -0.04),
        arrowstyle='->',
        color=ACCENT,
        linewidth=4,
        mutation_scale=45,
        transform=fig.transFigure,
        zorder=20
    )
    fig.patches.append(arrow_x)

    fig.text(
        0.50, -0.03,
        "Time",
        ha="center",
        va="bottom",
        fontsize=18,
        color=ACCENT
    )

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

def plot_shape_prediction_rollout_accuracy(
    agent1, agent2, labels, env,device,
    M=100, N_grounding_steps=5, K_rollout_steps=10,
):
    agents = [agent1, agent2]

    # Store full accuracies for mean & variance
    # Shape: [label][property] -> (M, K_rollout_steps)
    all_accuracies = {
        label: defaultdict(lambda: np.zeros((M, K_rollout_steps)))
        for label in labels
    }

    print(f"Starting {M} grounding and rollout experiments...")

    # === Main Loop ===
    for m in range(M):

        # Reset env + agents
        obs = env.reset()
        action = torch.distributions.OneHotCategorical(
            logits=torch.ones((1, 4), device=device)
        ).sample()

        for agent in agents:
            agent.dreamer.reset()
            agent.dreamer.state = agent.dreamer.step(action, obs)

        # Store ground-truth labels for the rollout
        true_labels = defaultdict(list)

        # === Grounding Phase ===
        for _ in range(N_grounding_steps):
            action = torch.distributions.OneHotCategorical(
                logits=torch.ones((1, 4), device=device)
            ).sample()
            obs = env.step(lambda o: action)[0]
            agent1.dreamer.state = agent1.dreamer.step(action, obs)
            agent2.dreamer.state = agent2.dreamer.step(action, obs)

        # === Imagination Rollout Phase ===
        agent1_state = agent1.dreamer.state.clone()
        agent2_state = agent2.dreamer.state.clone()

        for k in range(K_rollout_steps):

            # Environment step for true labels
            action = torch.distributions.OneHotCategorical(
                logits=torch.ones((1, 4), device=device)
            ).sample()
            obs = env.step(lambda o: action)[0]
            state = env._get_state()

            true_labels['size'].append(state["size_one_hot"].reshape((-1)))
            true_labels['color'].append(state["color_one_hot"].reshape((-1)))
            true_labels['shape'].append(state["shape_one_hot"].reshape((-1)))

            # True classes
            gt_size = true_labels['size'][-1].argmax(dim=-1)
            gt_color = true_labels['color'][-1].argmax(dim=-1)
            gt_shape = true_labels['shape'][-1].argmax(dim=-1)
            gt_labels = [gt_size, gt_color, gt_shape]

            # Rollout agents
            for i, agent in enumerate(agents):

                current_state = agent1_state if i == 0 else agent2_state

                next_state = agent.dreamer.dream_step(action, current_state)

                predicted_dists = agent.shape_predictor.predict(next_state.deter)

                properties = ['size', 'color', 'shape']

                for idx, prop in enumerate(properties):
                    predicted_class = predicted_dists[idx].probs.argmax(dim=-1).squeeze(0)
                    acc = (predicted_class == gt_labels[idx]).float().item()

                    # Store full accuracy (no averaging here)
                    all_accuracies[labels[i]][prop][m, k] = acc

                # Update rollout state
                if i == 0:
                    agent1_state = next_state
                else:
                    agent2_state = next_state

    print("Experiment complete. Plotting results...")

    # === Compute Mean + Variance ===
    properties = ['size', 'color', 'shape']

    mean_accuracies = {
        label: {
            prop: all_accuracies[label][prop].mean(axis=0)
            for prop in properties
        }
        for label in labels
    }

    var_accuracies = {
        label: {
            prop: all_accuracies[label][prop].var(axis=0)
            for prop in properties
        }
        for label in labels
    }

    # === Plotting ===
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    x_steps = np.arange(1, K_rollout_steps + 1)

    for prop_idx, prop in enumerate(properties):

        ax = axs[prop_idx]

        for label in labels:

            mean = mean_accuracies[label][prop]
            var = var_accuracies[label][prop]
            std = np.sqrt(var)

            # Mean curve
            ax.plot(
                x_steps, mean,
                label=label,
                marker='o', markersize=4
            )

            # Shaded ± std region
            ax.fill_between(
                x_steps,
                mean - std,
                mean + std,
                alpha=0.2
            )

        ax.set_title(prop.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xlabel("Imagination Step", fontsize=11)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        #ax.set_xticks(x_steps)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[0].set_ylabel("Accuracy", fontsize=11)

    # Legend outside
    handles, labels_list = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_list,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(labels)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # save in high quality:
    plt.show()
    

# NOTE: You must execute this function (and ensure 'device' and 'env' are defined)
# plot_shape_prediction_rollout_accuracy(agent1, agent2, env, M=100, N_grounding_steps=5, K_rollout_steps=10)


def generate_shape_env_overview(env):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    sns.set_theme(context="notebook", style="white", font_scale=1.2)
    ACCENT = "#1f77b4"   # modern blue

    num_colors = len(env.colors_list)
    num_shapes = len(env.shapes_list)
    num_sizes = len(env.sizes_list)
    num_shape_sizes = num_shapes * num_sizes

    # ---- Create main grid ----
    fig, axes = plt.subplots(
        num_colors, num_shape_sizes,
        figsize=(num_shape_sizes * 2.0, num_colors * 2.0),
    )

    if num_colors == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_shape_sizes == 1:
        axes = np.expand_dims(axes, axis=1)

    # ---- Plot images with NICE individual borders ----
    for i in range(num_colors):
        for j in range(num_shape_sizes):
            env.color_idx = i
            env.shape_size_idx = j
            img = env.generate_observation()

            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            border = FancyBboxPatch(
                            (0.02, 0.02),   # inset inside axes
                            0.96, 0.96,     # border spans almost entire image
                            boxstyle="round,pad=0.02,rounding_size=0.1",  # smooth, modern rounding
                            linewidth=3,
                            edgecolor="#1f77b4",   # or use agent color dynamically
                            facecolor="none",
                            alpha=0.9,
                            transform=ax.transAxes,   # IMPORTANT: consistent look across axes!
                            zorder=10
                        )
            ax.add_patch(border)

    fig.tight_layout(rect=[0.08, 0.10, 0.96, 0.95])


    # =========================================================
    #       OUTER BORDER THAT ALWAYS SHOWS IN JUPYTER
    # =========================================================

    overlay = fig.add_axes([0, 0, 1, 1], zorder=2)
    overlay.set_axis_off()



    # =========================================================
    #                DOUBLE-HEADED ARROWS
    # =========================================================

    # Vertical arrow (Color)
    arrow_y = FancyArrowPatch(
        posA=(0.065, 0.095),
        posB=(0.065, 0.945),
        arrowstyle='<->',
        color=ACCENT,
        linewidth=3,
        mutation_scale=45,
        transform=overlay.transAxes
    )
    overlay.add_patch(arrow_y)

    overlay.text(
        0.03, 0.515, "Color",
        ha="center", va="center",
        rotation="vertical",
        fontsize=18, fontweight="bold",
        color=ACCENT
    )

    # Horizontal arrow (Size/Shape)
    arrow_x = FancyArrowPatch(
        posA=(0.085, 0.075),
        posB=(0.94, 0.075),
        arrowstyle='<->',
        color=ACCENT,
        linewidth=3,
        mutation_scale=45,
        transform=overlay.transAxes
    )
    overlay.add_patch(arrow_x)

    overlay.text(
        0.515, 0.03, "Size / Shape",
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color=ACCENT
    )

    plt.show()
