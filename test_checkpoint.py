"""
Comprehensive checkpoint save/load testing for CTM PPO training.

Tests:
1. Basic save/load without errors
2. Training state (counters, metrics) restoration
3. Model weights preservation
4. Optimizer state preservation
"""

import torch
import os
import numpy as np
from torch import optim

from ctm.ctm_agent import CTMAgent
from ctm.ctm_rl import ContinuousThoughtMachineRL
from ctm.img_coder import MinesweeperConvEncoder
from environments.minesweeper.minesweeper import MinesweeperEnv
from train import save_model, load_model


def setup_models(device='cpu'):
    """Initialize agent, encoder, and optimizer with the same config as train.py."""
    torch.set_default_device(device)

    # Environment and encoder
    width, height, n_mines = 6, 6, 10
    env = MinesweeperEnv(width, height, n_mines)
    minesweeper_enc = MinesweeperConvEncoder(256, env.state_im.shape)

    # CTM and agent
    ctm = ContinuousThoughtMachineRL(
        iterations=5,
        d_model=2048,
        d_input=256,
        n_synch_out=64,
        synapse_depth=8,
        memory_length=25,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='minesweeper-backbone',
    )

    agent = CTMAgent(ctm=ctm, continuous_state_trace=True, device=device, num_actions=env.ntiles)

    # Optimizer
    all_params = list(agent.parameters()) + list(minesweeper_enc.parameters())
    optimizer = optim.Adam(all_params, lr=5e-4, eps=1e-5)

    return agent, minesweeper_enc, optimizer, env


def test_basic_save_load():
    """Test 1: Basic save and load without errors."""
    print("\n" + "="*60)
    print("TEST 1: Basic Save/Load")
    print("="*60)

    device = 'cpu'
    test_path = "./models/test_checkpoint.pt"

    # Setup
    agent, minesweeper_enc, optimizer, env = setup_models(device)

    # Create some training state
    global_step = 1234
    update = 10
    total_wins = 5
    episode_returns = [(100, 2.5), (200, 3.1), (300, 1.8)]
    episode_wins = [(100, 1), (200, 0), (300, 1)]
    update_logs = [
        {'step': 100, 'pg_loss': 0.5, 'v_loss': 0.3, 'entropy': 1.2, 'explained_variance': 0.7},
        {'step': 200, 'pg_loss': 0.4, 'v_loss': 0.25, 'entropy': 1.1, 'explained_variance': 0.75},
    ]

    # Save
    print("Saving checkpoint...")
    try:
        save_model(agent, minesweeper_enc, optimizer, global_step, update, total_wins,
                   episode_returns, episode_wins, update_logs, test_path)
        print(f"✓ Checkpoint saved to {test_path}")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False

    # Load into fresh models
    print("Loading checkpoint...")
    agent2, minesweeper_enc2, optimizer2, _ = setup_models(device)

    try:
        loaded_step, loaded_update, loaded_wins, loaded_returns, loaded_wins_list, loaded_logs = \
            load_model(agent2, minesweeper_enc2, optimizer2, test_path, device)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Loaded step: {loaded_step}, update: {loaded_update}, wins: {loaded_wins}")
    except Exception as e:
        print(f"✗ Load failed: {e}")
        return False

    print("✓ TEST 1 PASSED: Basic save/load works\n")
    return True


def test_training_state_restoration():
    """Test 2: Verify all training state is correctly restored."""
    print("\n" + "="*60)
    print("TEST 2: Training State Restoration")
    print("="*60)

    device = 'cpu'
    test_path = "./models/test_checkpoint.pt"

    # Setup
    agent, minesweeper_enc, optimizer, _ = setup_models(device)

    # Create specific training state to verify
    original_state = {
        'global_step': 4567,
        'update': 23,
        'total_wins': 42,
        'episode_returns': [(100, 2.5), (200, 3.1), (300, 1.8), (400, 4.2)],
        'episode_wins': [(100, 1), (200, 0), (300, 1), (400, 1)],
        'update_logs': [
            {'step': 100, 'pg_loss': 0.5, 'v_loss': 0.3, 'entropy': 1.2, 'explained_variance': 0.7},
            {'step': 200, 'pg_loss': 0.4, 'v_loss': 0.25, 'entropy': 1.1, 'explained_variance': 0.75},
            {'step': 300, 'pg_loss': 0.35, 'v_loss': 0.22, 'entropy': 1.0, 'explained_variance': 0.8},
        ]
    }

    # Save
    save_model(agent, minesweeper_enc, optimizer,
               original_state['global_step'],
               original_state['update'],
               original_state['total_wins'],
               original_state['episode_returns'],
               original_state['episode_wins'],
               original_state['update_logs'],
               test_path)

    # Load
    agent2, minesweeper_enc2, optimizer2, _ = setup_models(device)
    loaded_step, loaded_update, loaded_wins, loaded_returns, loaded_wins_list, loaded_logs = \
        load_model(agent2, minesweeper_enc2, optimizer2, test_path, device)

    # Verify each component
    all_passed = True

    if loaded_step == original_state['global_step']:
        print(f"✓ global_step restored correctly: {loaded_step}")
    else:
        print(f"✗ global_step mismatch: expected {original_state['global_step']}, got {loaded_step}")
        all_passed = False

    if loaded_update == original_state['update']:
        print(f"✓ update restored correctly: {loaded_update}")
    else:
        print(f"✗ update mismatch: expected {original_state['update']}, got {loaded_update}")
        all_passed = False

    if loaded_wins == original_state['total_wins']:
        print(f"✓ total_wins restored correctly: {loaded_wins}")
    else:
        print(f"✗ total_wins mismatch: expected {original_state['total_wins']}, got {loaded_wins}")
        all_passed = False

    if loaded_returns == original_state['episode_returns']:
        print(f"✓ episode_returns restored correctly ({len(loaded_returns)} entries)")
    else:
        print(f"✗ episode_returns mismatch")
        all_passed = False

    if loaded_wins_list == original_state['episode_wins']:
        print(f"✓ episode_wins restored correctly ({len(loaded_wins_list)} entries)")
    else:
        print(f"✗ episode_wins mismatch")
        all_passed = False

    if loaded_logs == original_state['update_logs']:
        print(f"✓ update_logs restored correctly ({len(loaded_logs)} entries)")
    else:
        print(f"✗ update_logs mismatch")
        all_passed = False

    if all_passed:
        print("✓ TEST 2 PASSED: All training state restored correctly\n")
    else:
        print("✗ TEST 2 FAILED: Some training state not restored correctly\n")

    return all_passed


def test_model_weights_preservation():
    """Test 3: Verify model weights produce identical outputs after loading."""
    print("\n" + "="*60)
    print("TEST 3: Model Weights Preservation")
    print("="*60)

    device = 'cpu'
    test_path = "./models/test_checkpoint.pt"

    # Setup
    agent, minesweeper_enc, optimizer, env = setup_models(device)

    # Perform a tiny bit of training to change weights from initialization
    env.reset()
    raw_obs = torch.from_numpy(env.state_im.T).float().unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        obs = minesweeper_enc(raw_obs)
        initial_state = agent.get_initial_state(1)
        done = torch.zeros(1).to(device)
        action, logprob, entropy, value, next_state, _, _, _ = agent.get_action_and_value(obs, initial_state, done)

    # Save current state
    save_model(agent, minesweeper_enc, optimizer, 100, 1, 0, [], [], [], test_path)

    # Record outputs with original model
    print("Generating outputs with original model...")
    with torch.no_grad():
        original_obs = minesweeper_enc(raw_obs)
        original_initial_state = agent.get_initial_state(1)
        original_action, original_logprob, original_entropy, original_value, _, _, _, _ = \
            agent.get_action_and_value(original_obs, original_initial_state, done)

    # Load into fresh models
    print("Loading checkpoint into fresh models...")
    agent2, minesweeper_enc2, optimizer2, _ = setup_models(device)
    load_model(agent2, minesweeper_enc2, optimizer2, test_path, device)

    # Generate outputs with loaded model
    print("Generating outputs with loaded model...")
    with torch.no_grad():
        loaded_obs = minesweeper_enc2(raw_obs)
        loaded_initial_state = agent2.get_initial_state(1)
        loaded_action, loaded_logprob, loaded_entropy, loaded_value, _, _, _, _ = \
            agent2.get_action_and_value(loaded_obs, loaded_initial_state, done)

    # Compare outputs
    all_passed = True
    tolerance = 1e-6

    # Check encoder outputs
    if torch.allclose(original_obs, loaded_obs, atol=tolerance):
        print(f"✓ Encoder outputs match (max diff: {(original_obs - loaded_obs).abs().max().item():.2e})")
    else:
        print(f"✗ Encoder outputs differ (max diff: {(original_obs - loaded_obs).abs().max().item():.2e})")
        all_passed = False

    # Check agent outputs
    if original_action.item() == loaded_action.item():
        print(f"✓ Actions match: {original_action.item()}")
    else:
        print(f"✗ Actions differ: {original_action.item()} vs {loaded_action.item()}")
        all_passed = False

    if torch.allclose(original_logprob, loaded_logprob, atol=tolerance):
        print(f"✓ Log probabilities match (diff: {(original_logprob - loaded_logprob).abs().item():.2e})")
    else:
        print(f"✗ Log probabilities differ (diff: {(original_logprob - loaded_logprob).abs().item():.2e})")
        all_passed = False

    if torch.allclose(original_value, loaded_value, atol=tolerance):
        print(f"✓ Values match (diff: {(original_value - loaded_value).abs().item():.2e})")
    else:
        print(f"✗ Values differ (diff: {(original_value - loaded_value).abs().item():.2e})")
        all_passed = False

    if all_passed:
        print("✓ TEST 3 PASSED: Model weights preserved correctly\n")
    else:
        print("✗ TEST 3 FAILED: Model outputs differ after loading\n")

    return all_passed


def test_optimizer_state_preservation():
    """Test 4: Verify optimizer state (momentum buffers) is preserved."""
    print("\n" + "="*60)
    print("TEST 4: Optimizer State Preservation")
    print("="*60)

    device = 'cpu'
    test_path = "./models/test_checkpoint.pt"

    # Setup
    agent, minesweeper_enc, optimizer, env = setup_models(device)

    # Perform one gradient update to create optimizer state (momentum buffers)
    env.reset()
    raw_obs = torch.from_numpy(env.state_im.T).float().unsqueeze(0).to(device)
    obs = minesweeper_enc(raw_obs)
    initial_state = agent.get_initial_state(1)
    done = torch.zeros(1).to(device)

    # Forward pass
    action, logprob, entropy, value, next_state, _, _, _ = agent.get_action_and_value(obs, initial_state, done)

    # Dummy loss and backward
    loss = -value.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Performed gradient update to initialize optimizer state")

    # Save
    save_model(agent, minesweeper_enc, optimizer, 100, 1, 0, [], [], [], test_path)

    # Load
    agent2, minesweeper_enc2, optimizer2, _ = setup_models(device)
    load_model(agent2, minesweeper_enc2, optimizer2, test_path, device)

    # Compare optimizer states
    print("Comparing optimizer states...")
    all_passed = True

    # Adam optimizer has 'exp_avg' and 'exp_avg_sq' for each parameter (momentum buffers)
    orig_state = optimizer.state_dict()['state']
    loaded_state = optimizer2.state_dict()['state']

    if len(orig_state) == len(loaded_state):
        print(f"✓ Optimizer tracks same number of parameters: {len(orig_state)}")
    else:
        print(f"✗ Optimizer state size mismatch: {len(orig_state)} vs {len(loaded_state)}")
        all_passed = False

    # Check a few parameter states
    num_params_checked = 0
    num_params_match = 0

    for param_id in list(orig_state.keys())[:5]:  # Check first 5 params
        if param_id not in loaded_state:
            print(f"✗ Parameter {param_id} missing in loaded optimizer")
            all_passed = False
            continue

        orig_exp_avg = orig_state[param_id]['exp_avg']
        loaded_exp_avg = loaded_state[param_id]['exp_avg']

        if torch.allclose(orig_exp_avg, loaded_exp_avg, atol=1e-6):
            num_params_match += 1
        else:
            print(f"✗ Momentum buffer mismatch for param {param_id}")
            all_passed = False

        num_params_checked += 1

    if num_params_match == num_params_checked:
        print(f"✓ All checked momentum buffers match ({num_params_checked}/{num_params_checked})")
    else:
        print(f"✗ Only {num_params_match}/{num_params_checked} momentum buffers match")
        all_passed = False

    # Check learning rate
    orig_lr = optimizer.param_groups[0]['lr']
    loaded_lr = optimizer2.param_groups[0]['lr']

    if orig_lr == loaded_lr:
        print(f"✓ Learning rate preserved: {orig_lr}")
    else:
        print(f"✗ Learning rate mismatch: {orig_lr} vs {loaded_lr}")
        all_passed = False

    if all_passed:
        print("✓ TEST 4 PASSED: Optimizer state preserved correctly\n")
    else:
        print("✗ TEST 4 FAILED: Optimizer state not fully preserved\n")

    return all_passed


def main():
    """Run all checkpoint tests."""
    print("\n" + "="*60)
    print("CTM PPO CHECKPOINT TEST SUITE")
    print("="*60)

    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)

    # Run all tests
    results = []
    results.append(("Basic Save/Load", test_basic_save_load()))
    results.append(("Training State Restoration", test_training_state_restoration()))
    results.append(("Model Weights Preservation", test_model_weights_preservation()))
    results.append(("Optimizer State Preservation", test_optimizer_state_preservation()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")

    # Cleanup
    if os.path.exists("./models/test_checkpoint.pt"):
        os.remove("./models/test_checkpoint.pt")
        print("Cleaned up test checkpoint file")

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
