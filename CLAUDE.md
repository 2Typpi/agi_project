# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run PPO training
python train.py

# Launch Jupyter notebooks
jupyter lab

# Run switchboard environment with pygame UI
python environments/switchboard/pygame_interface.py --scenario direct_rules

# Run shape environment with pygame UI
python environments/shape_environment/pygame_interface.py
```

There is no build system, test suite, or linter configured in this project.

## Architecture

### CTM for Reinforcement Learning

The project applies a **Continuous Thought Machine (CTM)** to Minesweeper via PPO. The CTM is a biologically-inspired recurrent architecture with three distinguishing ideas:

1. **Internal recurrence** (`iterations` ticks per forward pass): Synapses combine the current input with the last post-activation state, writing pre-activations into a sliding trace buffer.
2. **Neuron-Level Models (NLMs)**: Per-neuron private MLPs that process each neuron's own trace history to produce post-activations — each neuron has memory of its temporal dynamics.
3. **Synchronisation as representation**: The actor/critic heads consume a "synchronisation" signal computed as weighted dot-products between pairs of selected neurons' post-activation histories. Output size is determined by `n_synch_out`.

### Key files

| File | Role |
|---|---|
| `train.py` | PPO training loop: rollout collection → GAE advantage estimation → minibatch update |
| `ctm/ctm.py` | Base `ContinuousThoughtMachine` nn.Module with full attention support |
| `ctm/ctm_rl.py` | `ContinuousThoughtMachineRL` — strips attention, adds learnable `start_activated_trace`, overrides backbone to `MinesweeperConvEncoder` |
| `ctm/ctm_agent.py` | `CTMAgent` — wraps CTM + actor/critic heads; handles per-episode state masking and `get_action_and_value()` |
| `ctm/img_coder.py` | `MinesweeperConvEncoder`: Conv2d(1→32, k=3, pad=1) → Flatten → Linear → latent |
| `ctm/action_head.py` | `UniversalActionHead`: Linear → ReLU → Linear → logits |
| `ctm/critic_head.py` | `UniversalCriticHead`: Linear → ReLU → Linear → scalar value |
| `ctm/modules.py` | `SynapseUNET`, NLM implementations, `MiniGridBackbone`, `ClassicControlBackbone` |
| `environments/minesweeper/minesweeper.py` | `MinesweeperEnv`: state is `(H, W, 1)` float array scaled by `/8`; rewards: win=+1, lose=−1, progress=+0.3, guess=−0.3, no_progress=−0.3 |

### PPO training data flow

1. `MinesweeperConvEncoder` (`minesweeper_enc`) encodes `(1, 1, H, W)` raw state → `(1, latent_dim)`. **This encoder must be in the optimizer** and raw observations must be stored so the encoder is re-called (with gradient) during the learning phase.
2. `CTMAgent.get_action_and_value(latent, ctm_state, done)` runs the CTM recurrence for `iterations` ticks, computes synchronisation, then passes through actor/critic heads.
3. CTM hidden state = `(state_trace, activated_state_trace)`, both shape `(num_envs, d_model, memory_length)`. Episode resets are handled by masking the previous state with `(1 - done)` and adding the learned initial state scaled by `done`.
4. During the learning phase, hidden state is selected per-env as `initial_state[0][mbenvinds]` (index dim 0 = envs), **not** `[:, mbenvinds, :]` which would incorrectly index the `d_model` axis.

### CTM backbone note

`ContinuousThoughtMachineRL` registers a `self.backbone = MinesweeperConvEncoder(...)` as a submodule (included in `ctm.parameters()`), but the comment in `ctm_rl.py:138` documents that the backbone is intentionally called *externally* in the training loop. The external `minesweeper_enc` in `train.py` is the one actually used — it must be kept in sync with the optimizer.

### Other RL algorithms (`rl/`)

These were used in prior coursework exercises and are not part of the active CTM training:
- `gcsl.py` — Goal-Conditioned Supervised Learning with goal relabeling
- `crl.py` — Stable Contrastive RL with bilinear Q-function
- `a2c.py` — Advantage Actor-Critic with lambda returns
- `rl/dreamer/` — DreamerV3-style world model with RSSM
