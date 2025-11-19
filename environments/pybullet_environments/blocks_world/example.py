import os
import sys
import torch
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from environments.pybullet_environments.blocks_world import BlocksWorld


if __name__ == "__main__":

    # Create environment

    env = BlocksWorld(gui=False, image_width=32, image_height=32, 
                                   time_scaling=0, dt=1/120, reset_objects=True, reset_agent=True, randomize_object_positions=True)
    
    # Define a simple benchmark policy (random movement)
    def benchmark_policy(obs):
        time.sleep(0.01)  # 100ms computation time
        # Random movement
        forward = (torch.rand(1, device=obs.device) - 0.5) * 2  # [-1, 1]
        rotation = (torch.rand(1, device=obs.device) - 0.5) * 2  # [-1, 1]
        return torch.tensor([forward, rotation], device=obs.device).squeeze()
    
    # Benchmark the system
    print("Benchmarking policy...")
    policy_time = env.benchmark_policy(benchmark_policy, num_trials=3)
    print(f"Benchmark policy time: {policy_time:.2f} ms")
    
    print("Benchmarking simulation...")
    sim_time = env.benchmark_simulation(num_trials=30)
    print(f"Simulation step time: {sim_time:.4f} ms")
    
    # Test with different policy speeds
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")  # Should be [3, 128, 128]
    try:
        # Run some episodes
        for episode in range(100):
            print(f"\n=== Episode {episode + 1} ===")
            obs = env.reset()
            
            for step in range(100):
                # Create a policy with variable computation time
                def variable_policy(obs):
                    # Simulate variable computation (50ms to 200ms)
                    computation_time = 1#0.05 + torch.rand(1).item() * 0.15
                    time.sleep(computation_time)
                    
                    # Simple movement policy: move forward and turn slightly
                    forward = 1.0 + (torch.rand(1, device=obs.device) - 0.5) * 0.5
                    rotation = (torch.rand(1, device=obs.device) - 0.5) * 0.5
                    
                    return torch.tensor([forward, rotation], device=obs.device).squeeze()
                
                # Step environment
                obs_list, info = env.step(variable_policy)
                final_obs = obs_list[-1]
                
                # Print info every 10 steps
                if step % 10 == 0:
                    print(f"Step {step}:")
                    print(f"  Policy time: {info['actual_policy_time_ms']:.1f} ms")
                    print(f"  Environment steps: {info['num_environment_steps']}")
                    print(f"  Observations received: {len(obs_list)}")
                    print(f"  Final observation shape: {final_obs.shape}")
                    
                # Small delay for visualization
                #time.sleep(0.02)
        
    except KeyboardInterrupt:
        print("\\nStopped by user")
    
    finally:
        env.close()
        print("Environment closed")