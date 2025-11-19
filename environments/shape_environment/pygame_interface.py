"""
Pygame interface for ShapeEnv environment.

Provides a visual interface to interact with shapes:
- Arrow keys to change colors and shapes
- Real-time visual feedback
- Clean, modern interface
"""

import os
import sys
import torch
import pygame
import time
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.shape_environment.shape_env import ShapeEnv


class ShapeEnvPygameInterface:
    """
    Pygame visual interface for ShapeEnv environment.

    Controls:
    - LEFT/RIGHT Arrow: Change color (action 0/1)
    - UP/DOWN Arrow: Change shape (action 2/3)
    - Press R to reset environment
    - Press Q or ESC to quit
    - Press P to pause/resume auto-stepping
    - Press ENTER for single step (when paused)
    - Press +/- to adjust speed
    """

    def __init__(
        self,
        env: ShapeEnv,
        width: int = 1000,
        height: int = 700,
        fps: int = 60,
    ):
        """
        Args:
            env: ShapeEnv environment
            width: Window width
            height: Window height
            fps: Target frames per second
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Shape Environment Interface")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.tiny_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 48)
        self.large_font = pygame.font.Font(None, 72)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_GRAY = (200, 200, 200)
        self.RED = (255, 60, 60)
        self.GREEN = (60, 255, 100)
        self.BLUE = (80, 140, 255)
        self.YELLOW = (255, 220, 0)
        self.ORANGE = (255, 140, 0)
        self.PURPLE = (180, 100, 255)
        self.CYAN = (0, 200, 255)
        self.BG_DARK = (20, 20, 30)
        self.PANEL_BG = (30, 30, 45)

        # Layout parameters
        self.margin = 20
        self.panel_width = 300  # Right panel for info

        # Track arrow key states
        self.arrow_keys_pressed = {
            'left': False,
            'right': False,
            'up': False,
            'down': False
        }

        # Stepping control
        self.paused = False  # Start running (like switchboard)
        self.auto_step_speed = 10  # Steps per second when auto-stepping
        self.time_accumulator = 0.0
        self.steps_since_last_update = 0

        # Performance tracking
        self.step_times = []
        self.max_step_history = 100
        self.last_step_count = 0
        self.last_step_time = None
        self.actual_steps_per_sec = 0.0

        # Color names mapping
        self.color_names = {
            'r': 'Red',
            'g': 'Green',
            'b': 'Blue',
            'k': 'Black'
        }

        # Shape names
        self.shape_names = ['Circle', 'Triangle', 'Square', 'Pentagon']
        self.size_names = ['Small', 'Medium', 'Large']

    def _get_action_from_keys(self) -> torch.Tensor:
        """Convert arrow key states to action tensor"""
        # Priority: if multiple keys pressed, use first one found
        if self.arrow_keys_pressed['left']:
            return torch.tensor([0])  # color-
        elif self.arrow_keys_pressed['right']:
            return torch.tensor([1])  # color+
        elif self.arrow_keys_pressed['down']:
            return torch.tensor([2])  # shape-
        elif self.arrow_keys_pressed['up']:
            return torch.tensor([3])  # shape+
        else:
            return torch.tensor([4])  # no-op (position changes)

    def _draw_header(self):
        """Draw header with title and status info"""
        # Title
        title = self.title_font.render("SHAPE ENVIRONMENT", True, self.CYAN)
        self.screen.blit(title, (self.margin * 2, self.margin))

        # Stepping mode
        status_x = self.width - self.panel_width - 250
        status_y = self.margin + 10

        step_mode = "PAUSED" if self.paused else "RUNNING"
        step_color = self.RED if self.paused else self.GREEN
        step_mode_text = self.small_font.render(f"Mode: {step_mode}", True, step_color)
        self.screen.blit(step_mode_text, (status_x, status_y))

        # Speed indicator (when not paused)
        if not self.paused:
            target_speed = self.auto_step_speed
            actual_speed = self.actual_steps_per_sec

            # Color code based on whether we're hitting target
            if actual_speed >= target_speed * 0.9:
                speed_color = self.GREEN
            elif actual_speed >= target_speed * 0.7:
                speed_color = self.YELLOW
            else:
                speed_color = self.RED

            speed_text = self.tiny_font.render(
                f"Speed: {actual_speed:.1f}/{target_speed} steps/sec",
                True, speed_color
            )
            self.screen.blit(speed_text, (status_x, status_y + 20))

        # Step counter
        step_text = self.font.render(f"STEP: {self.env.step_count}", True, self.WHITE)
        self.screen.blit(step_text, (status_x + 180, self.margin + 10))

        # Draw separator line
        line_y = 80
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (self.margin, line_y),
                        (self.width - self.panel_width - self.margin, line_y), 2)

    def _draw_shape_display(self):
        """Draw the current shape in the center of the screen"""
        # Calculate display area (left side, excluding panel)
        display_width = self.width - self.panel_width - self.margin * 3
        display_height = self.height - 200
        display_x = self.margin * 2
        display_y = 100

        # Draw background panel
        display_rect = pygame.Rect(display_x, display_y, display_width, display_height)
        pygame.draw.rect(self.screen, self.PANEL_BG, display_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.CYAN, display_rect, 2, border_radius=10)

        # Get current state as numpy array (convert from CHW to HWC)
        state_tensor = self.env.state.cpu()
        state_np = state_tensor.permute(1, 2, 0).numpy()

        # Convert to RGB (assume RGBA, drop alpha or convert)
        if state_np.shape[2] == 4:
            # RGBA to RGB
            state_rgb = state_np[:, :, :3]
        else:
            state_rgb = state_np

        # Scale to 0-255 range
        state_rgb = (state_rgb * 255).astype(np.uint8)

        # Create pygame surface from numpy array
        # Resize to fit display area while maintaining aspect ratio
        shape_h, shape_w = state_rgb.shape[:2]

        # Calculate scaling to fit in display area
        max_display_size = min(display_width - 40, display_height - 40)
        scale = max_display_size / max(shape_h, shape_w)

        new_w = int(shape_w * scale)
        new_h = int(shape_h * scale)

        # Create surface and scale it
        shape_surface = pygame.surfarray.make_surface(np.transpose(state_rgb, (1, 0, 2)))
        shape_surface = pygame.transform.scale(shape_surface, (new_w, new_h))

        # Center the shape in display area
        shape_x = display_x + (display_width - new_w) // 2
        shape_y = display_y + (display_height - new_h) // 2

        # Draw white background behind shape
        bg_rect = pygame.Rect(shape_x - 10, shape_y - 10, new_w + 20, new_h + 20)
        pygame.draw.rect(self.screen, self.WHITE, bg_rect, border_radius=5)

        # Draw the shape
        self.screen.blit(shape_surface, (shape_x, shape_y))

    def _get_shape_info(self):
        """Get current shape information"""
        color_idx = self.env.color_count % len(self.env.color)
        color_code = self.env.color[color_idx]
        color_name = self.color_names.get(color_code, color_code)

        shape_idx = self.env.shape_count % (len(self.env.shape) * len(self.env.r))
        size_idx = shape_idx % len(self.env.r)
        shape_type_idx = shape_idx // len(self.env.r)

        shape_name = self.shape_names[shape_type_idx]
        size_name = self.size_names[size_idx]

        return color_name, shape_name, size_name, color_idx, shape_type_idx, size_idx

    def _draw_info_panel(self):
        """Draw right panel showing current shape info and controls"""
        panel_x = self.width - self.panel_width
        panel_y = 0

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, self.DARK_GRAY, (panel_x, 0), (panel_x, self.height), 2)

        # Panel title
        title = self.font.render("SHAPE INFO", True, self.CYAN)
        self.screen.blit(title, (panel_x + 20, 20))

        y_offset = 70

        # Get shape info
        color_name, shape_name, size_name, color_idx, shape_type_idx, size_idx = self._get_shape_info()

        # Display current state
        info_items = [
            ("Color:", color_name, self.GREEN),
            ("Shape:", shape_name, self.BLUE),
            ("Size:", size_name, self.YELLOW),
        ]

        for label, value, color in info_items:
            # Label
            label_text = self.small_font.render(label, True, self.GRAY)
            self.screen.blit(label_text, (panel_x + 20, y_offset))

            # Value
            value_text = self.font.render(value, True, color)
            self.screen.blit(value_text, (panel_x + 20, y_offset + 22))

            y_offset += 65

        # Draw indices info
        y_offset += 20
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 15, y_offset),
                        (panel_x + self.panel_width - 15, y_offset), 1)
        y_offset += 20

        indices_text = self.small_font.render("INDICES", True, self.CYAN)
        self.screen.blit(indices_text, (panel_x + 20, y_offset))
        y_offset += 30

        # Color index
        color_text = self.tiny_font.render(f"Color: {self.env.color_count} (mod {color_idx})", True, self.LIGHT_GRAY)
        self.screen.blit(color_text, (panel_x + 20, y_offset))
        y_offset += 22

        # Shape index
        shape_text = self.tiny_font.render(f"Shape: {self.env.shape_count} (mod {shape_type_idx * len(self.env.r) + size_idx})", True, self.LIGHT_GRAY)
        self.screen.blit(shape_text, (panel_x + 20, y_offset))
        y_offset += 40

        # Draw controls info at bottom
        controls_y = self.height - 320
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 15, controls_y),
                        (panel_x + self.panel_width - 15, controls_y), 1)

        controls_title = self.small_font.render("CONTROLS", True, self.CYAN)
        self.screen.blit(controls_title, (panel_x + 20, controls_y + 15))

        # Controls list
        controls = [
            "← →: Change color",
            "↑ ↓: Change shape",
            "P: Pause/Resume",
            "ENTER: Single step",
            "+/-: Speed",
            "R: Reset",
            "Q/ESC: Quit",
        ]

        cy = controls_y + 50
        for control in controls:
            text = self.small_font.render(control, True, self.LIGHT_GRAY)
            self.screen.blit(text, (panel_x + 20, cy))
            cy += 25

        # Draw current action indicator
        action_y = self.height - 50
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 15, action_y),
                        (panel_x + self.panel_width - 15, action_y), 1)

        action_label = self.tiny_font.render("CURRENT ACTION:", True, self.GRAY)
        self.screen.blit(action_label, (panel_x + 20, action_y + 10))

        # Show which key is pressed
        action_text = "No-op (position changes)"
        action_color = self.GRAY

        if self.arrow_keys_pressed['left']:
            action_text = "← Color -"
            action_color = self.ORANGE
        elif self.arrow_keys_pressed['right']:
            action_text = "→ Color +"
            action_color = self.GREEN
        elif self.arrow_keys_pressed['down']:
            action_text = "↓ Shape -"
            action_color = self.PURPLE
        elif self.arrow_keys_pressed['up']:
            action_text = "↑ Shape +"
            action_color = self.CYAN

        action_display = self.small_font.render(action_text, True, action_color)
        self.screen.blit(action_display, (panel_x + 20, action_y + 28))

    def run(self):
        """Main game loop"""
        running = True

        # Run benchmarks first
        print("Running environment benchmarks...")

        def benchmark_policy(obs):
            return torch.tensor([0])

        self.env.benchmark_policy(benchmark_policy, num_trials=5)
        self.env.benchmark_simulation(num_trials=10)

        obs = self.env.reset()

        print("Pygame interface started!")
        print(f"Environment: {self.env.lim}x{self.env.lim} pixels")
        print(f"Colors: {self.env.color}")
        print(f"Shapes: {self.env.shape}")

        while running:
            frame_start = time.time()
            dt = 1.0 / self.fps  # Time delta in seconds

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_r:
                        obs = self.env.reset()
                        print("Environment reset")

                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                        print(f"Stepping: {'PAUSED' if self.paused else 'RUNNING'}")

                    elif event.key == pygame.K_RETURN:
                        # Single step when paused
                        if self.paused:
                            self.steps_since_last_update = 1
                            print("Single step")

                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        # Increase speed
                        self.auto_step_speed = min(60, self.auto_step_speed + 1)
                        print(f"Speed: {self.auto_step_speed} steps/sec")

                    elif event.key == pygame.K_MINUS:
                        # Decrease speed
                        self.auto_step_speed = max(1, self.auto_step_speed - 1)
                        print(f"Speed: {self.auto_step_speed} steps/sec")

                    # Arrow keys - hold to apply action
                    elif event.key == pygame.K_LEFT:
                        self.arrow_keys_pressed['left'] = True
                    elif event.key == pygame.K_RIGHT:
                        self.arrow_keys_pressed['right'] = True
                    elif event.key == pygame.K_UP:
                        self.arrow_keys_pressed['up'] = True
                    elif event.key == pygame.K_DOWN:
                        self.arrow_keys_pressed['down'] = True

                elif event.type == pygame.KEYUP:
                    # Release arrow keys
                    if event.key == pygame.K_LEFT:
                        self.arrow_keys_pressed['left'] = False
                    elif event.key == pygame.K_RIGHT:
                        self.arrow_keys_pressed['right'] = False
                    elif event.key == pygame.K_UP:
                        self.arrow_keys_pressed['up'] = False
                    elif event.key == pygame.K_DOWN:
                        self.arrow_keys_pressed['down'] = False

            # Determine how many steps to take this frame
            num_steps_this_frame = 0

            if not self.paused:
                # Auto-stepping based on speed
                self.time_accumulator += dt
                steps_per_second = self.auto_step_speed
                time_per_step = 1.0 / steps_per_second

                # Calculate how many steps to take this frame
                while self.time_accumulator >= time_per_step:
                    self.time_accumulator -= time_per_step
                    num_steps_this_frame += 1

            if self.steps_since_last_update > 0:
                num_steps_this_frame += self.steps_since_last_update
                self.steps_since_last_update = 0

            # Step environment multiple times if needed (like switchboard)
            if num_steps_this_frame > 0:
                for _ in range(num_steps_this_frame):
                    # Get action from arrow keys
                    action = self._get_action_from_keys()

                    # Step the environment with RealTimeEnvironment timing
                    def policy_fn(_obs):
                        return action

                    obs_list, info = self.env.step(policy_fn)
                    obs = obs_list[-1]

            # Calculate actual steps per second
            current_time = time.time()
            if self.last_step_time is not None and not self.paused:
                time_elapsed = current_time - self.last_step_time
                if time_elapsed > 0.5:  # Update every 0.5 seconds
                    steps_taken = self.env.step_count - self.last_step_count
                    self.actual_steps_per_sec = steps_taken / time_elapsed
                    self.last_step_count = self.env.step_count
                    self.last_step_time = current_time
            elif self.last_step_time is None:
                self.last_step_time = current_time
                self.last_step_count = self.env.step_count

            # Draw everything
            self.screen.fill(self.BG_DARK)

            # Draw header
            self._draw_header()

            # Draw shape display
            self._draw_shape_display()

            # Draw info panel
            self._draw_info_panel()

            # Update display
            pygame.display.flip()

            # Track performance
            frame_time = (time.time() - frame_start) * 1000  # Convert to ms
            self.step_times.append(frame_time)
            if len(self.step_times) > self.max_step_history:
                self.step_times.pop(0)

            # Control frame rate
            self.clock.tick(self.fps)

        pygame.quit()
        print("Interface closed")


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run ShapeEnv with Pygame interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python pygame_interface.py

  # Run with noise enabled
  python pygame_interface.py --noisy --noise_value 0.2

  # Run with custom window size
  python pygame_interface.py --width 1200 --height 800
        """
    )

    parser.add_argument(
        '--noisy',
        action='store_true',
        help='Enable noise in shape generation'
    )

    parser.add_argument(
        '--noise_value',
        type=float,
        default=0.1,
        help='Noise value (default: 0.1)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=1000,
        help='Window width (default: 1000)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=700,
        help='Window height (default: 700)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Target FPS (default: 60)'
    )

    args = parser.parse_args()

    # Create environment
    # Use low time_scaling to ensure 1 policy call = 1 env step (no timing variance)
    env = ShapeEnv(
        noisy=args.noisy,
        noise_value=args.noise_value,
        device='cpu',
        time_scaling=0.0  # Low scaling prevents multiple steps per policy call
    )

    print("=" * 60)
    print("SHAPE ENVIRONMENT PYGAME INTERFACE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Noisy: {args.noisy}")
    print(f"  Noise value: {args.noise_value}")
    print(f"  Window size: {args.width}x{args.height}")
    print(f"  Colors: {env.color}")
    print(f"  Shapes: {env.shape}")
    print(f"  Sizes: {env.r}")

    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("  Navigation (hold keys):")
    print("    - LEFT/RIGHT Arrow: Change color")
    print("    - UP/DOWN Arrow: Change shape")
    print("    - No keys: Environment still progresses (position changes)")
    print("  Stepping:")
    print("    - P: Pause/Resume auto-stepping")
    print("    - ENTER: Single step (when paused)")
    print("    - +/-: Adjust speed")
    print("  Other:")
    print("    - R: Reset environment")
    print("    - Q/ESC: Quit")
    print("\nStarting RUNNING - shapes animate even without input!")
    print("=" * 60)
    print()

    # Create pygame interface
    interface = ShapeEnvPygameInterface(
        env,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    # Run the interface
    interface.run()
