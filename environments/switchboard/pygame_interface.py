"""
Pygame interface for Switchboard environment.

Provides a visual interface to interact with the switchboard:
- Click buttons to activate them
- See observation slots light up based on rules
- Real-time visual feedback
"""

import os
import sys
import torch
import pygame
import time
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.switchboard.switchboard import Switchboard, Rule


class SwitchboardPygameInterface:
    """
    Pygame visual interface for Switchboard environment.

    Controls:
    - Click buttons to press them
    - Press R to reset environment
    - Press Q or ESC to quit
    - Press SPACE to toggle auto-policy
    """

    def __init__(
        self,
        env: Switchboard,
        width: int = 1200,
        height: int = 700,
        fps: int = 60,
        auto_policy: Optional[callable] = None
    ):
        """
        Args:
            env: Switchboard environment
            width: Window width
            height: Window height
            fps: Target frames per second
            auto_policy: Optional policy function(obs) -> action for auto mode
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_policy = auto_policy
        self.auto_mode = False

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Switchboard Interface")
        self.clock = pygame.time.Clock()
        self._calculate_layout()

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

        # Track button states
        self.button_pressed = [False] * env.action_dim
        self.mouse_down_on_button = None
        self.mouse_held_buttons = set()  # Track which buttons are held by mouse

        # Stepping control
        self.paused = False  # Start paused for manual control
        self.auto_step_speed = 30  # Steps per second when auto-stepping
        self.steps_since_last_update = 0
        self.time_accumulator = 0.0

        # UI control
        self.show_rules_panel = False  # Toggle rules panel visibility
        self.rules_scroll_offset = 0  # Scroll offset for rules list
        self.tooltip_text = None

        # Keyboard mappings (0-9 keys map to buttons 0-9)
      # The new, robust mapping
        self.key_to_button = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        }

        # Performance tracking
        self.step_times = []
        self.max_step_history = 100
        self.last_step_count = 0
        self.last_step_time = None

    def _calculate_layout(self):
        """Calculate layout parameters based on window size."""
        # Proportional scaling factor (based on default size 1200x700)
        scale_w = self.width / 1200
        scale_h = self.height / 700
        scale = min(scale_w, scale_h)

        # Layout parameters
        self.button_radius = int(35 * scale)
        self.slot_width = int(50 * scale)
        self.slot_height = int(100 * scale)
        self.margin = int(20 * scale)
        self.panel_width = int(350 * scale_w)  # Panel scales with width

        # Font sizes
        self.font = pygame.font.Font(None, int(24 * scale))
        self.small_font = pygame.font.Font(None, int(18 * scale))
        self.tiny_font = pygame.font.Font(None, int(14 * scale))
        self.title_font = pygame.font.Font(None, int(42 * scale))

    def _get_button_positions(self) -> List[tuple]:
        """Calculate button positions in a grid layout"""
        positions = []
        # Use area excluding right panel (if visible)
        panel_width = self.panel_width if self.show_rules_panel else 0
        usable_width = self.width - panel_width - self.margin * 3

        cols = min(5, self.env.action_dim)

        start_x = self.margin * 2
        start_y = self.height // 6
        spacing_x = usable_width // cols
        spacing_y = self.height // 7

        for i in range(self.env.action_dim):
            row = i // cols
            col = i % cols
            x = start_x + col * spacing_x + spacing_x // 2
            y = start_y + row * spacing_y
            positions.append((x, y))

        return positions

    def _get_slot_positions(self) -> List[tuple]:
        """Calculate observation slot positions"""
        positions = []
        # Use area excluding right panel (if visible)
        panel_width = self.panel_width if self.show_rules_panel else 0
        usable_width = self.width - panel_width - self.margin * 3

        start_x = self.margin * 2
        start_y = self.height - self.height // 5
        spacing = usable_width // max(self.env.obs_dim, 1)

        for i in range(self.env.obs_dim):
            x = start_x + i * spacing + spacing // 2 - self.slot_width // 2
            y = start_y
            positions.append((x, y))

        return positions

    def _draw_button(self, pos: tuple, index: int, is_pressed: bool):
        """Draw a button with modern styling"""
        x, y = pos

        # Button appearance with glow effect
        if is_pressed:
            # Draw glow
            for i in range(3):
                glow_radius = self.button_radius + 8 - i * 3
                glow_alpha = 50 - i * 15
                glow_color = (*self.ORANGE[:3], glow_alpha)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (x - glow_radius, y - glow_radius))

            color = self.ORANGE
            radius = self.button_radius - 3
            border_color = self.YELLOW
        else:
            color = self.BLUE
            radius = self.button_radius
            border_color = self.CYAN

        # Draw button circle with gradient effect (simulated with multiple circles)
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, border_color, (x, y), radius, 3)

        # Draw button label
        label = self.font.render(str(index), True, self.WHITE)
        label_rect = label.get_rect(center=(x, y))
        self.screen.blit(label, label_rect)

        # Draw "Button" text below
        button_text = self.tiny_font.render(f"BTN", True, self.LIGHT_GRAY)
        text_rect = button_text.get_rect(center=(x, y + self.button_radius + 12))
        self.screen.blit(button_text, text_rect)

    def _draw_slot(self, pos: tuple, index: int, value: float):
        """Draw an observation slot with modern styling"""
        x, y = pos

        # Background with rounded corners (simulated)
        bg_rect = pygame.Rect(x, y, self.slot_width, self.slot_height)
        pygame.draw.rect(self.screen, self.DARK_GRAY, bg_rect, border_radius=5)

        # Fill based on activation value
        if value > 0.01:
            fill_height = int((self.slot_height - 4) * value)
            fill_y = y + self.slot_height - fill_height - 2

            # Color gradient based on value
            if value > 0.7:
                color = self.GREEN
            elif value > 0.4:
                color = self.YELLOW
            else:
                color = self.ORANGE

            fill_rect = pygame.Rect(x + 2, fill_y, self.slot_width - 4, fill_height)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

            # Add glow effect for high activation
            if value > 0.8:
                glow_surface = pygame.Surface((self.slot_width + 10, self.slot_height + 10), pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, (*color, 30), (0, 0, self.slot_width + 10, self.slot_height + 10),
                               border_radius=8)
                self.screen.blit(glow_surface, (x - 5, y - 5))

        # Draw border
        pygame.draw.rect(self.screen, self.CYAN, bg_rect, 2, border_radius=5)

        # Draw slot index at top
        label = self.small_font.render(str(index), True, self.WHITE)
        label_rect = label.get_rect(center=(x + self.slot_width // 2, y - 18))
        self.screen.blit(label, label_rect)

        # Draw "SLOT" text
        slot_text = self.tiny_font.render("SLOT", True, self.GRAY)
        slot_rect = slot_text.get_rect(center=(x + self.slot_width // 2, y - 6))
        self.screen.blit(slot_text, slot_rect)

        # Draw value text below
        value_text = self.small_font.render(f"{value:.2f}", True, self.WHITE if value > 0.1 else self.GRAY)
        value_rect = value_text.get_rect(center=(x + self.slot_width // 2, y + self.slot_height + 18))
        self.screen.blit(value_text, value_rect)

    def _draw_header(self):
        """Draw header with title and controls info"""
        # Title
        title = self.title_font.render("SWITCHBOARD", True, self.CYAN)
        self.screen.blit(title, (self.margin * 2, self.margin))

        # Mode and stepping status
        status_x = self.width - self.panel_width - 280

        # Auto-policy mode (only show if auto_policy is available)
        status_y = self.margin + 5
        if self.auto_policy:
            mode = "AUTO" if self.auto_mode else "MANUAL"
            mode_color = self.GREEN if self.auto_mode else self.ORANGE
            mode_text = self.small_font.render(f"Policy: {mode}", True, mode_color)
            self.screen.blit(mode_text, (status_x, status_y))
            status_y += 20

        # Stepping mode
        step_mode = "PAUSED" if self.paused else "RUNNING"
        step_color = self.RED if self.paused else self.GREEN
        step_mode_text = self.small_font.render(f"Stepping: {step_mode}", True, step_color)
        self.screen.blit(step_mode_text, (status_x, status_y))

        # Speed indicator (when not paused)
        if not self.paused:
            target_speed = self.auto_step_speed


            speed_text = self.tiny_font.render(
                f"Target speed: {target_speed} steps/sec",
                True, self.GREEN
            )
            self.screen.blit(speed_text, (status_x, status_y + 20))

        # Step counter
        step_text = self.font.render(f"STEP: {self.env.step_count}", True, self.WHITE)
        self.screen.blit(step_text, (status_x + 170, self.margin + 10))

        # Draw separator line
        line_y = 75
        panel_width = self.panel_width if self.show_rules_panel else 0
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (self.margin, line_y),
                        (self.width - panel_width - self.margin, line_y), 2)

    def _check_rule_condition_met(self, rule, action, obs):
        """
        Check if a rule is active by evaluating it and checking for a non-zero update.
        This is the most reliable way to determine if a rule has an effect.
        """
        # try:
        #     # Ensure action and obs are tensors
        #     if not isinstance(action, torch.Tensor):
        #         action = torch.tensor(action, dtype=torch.float32)
        #     if not isinstance(obs, torch.Tensor):
        #         obs = torch.tensor(obs, dtype=torch.float32)

        #     # Evaluate the rule to get the potential observation update
        #     obs_update = rule.evaluate(action, obs)

        #     # A rule is active if its update has any non-zero values
        #     return torch.any(obs_update > 0)
        # except Exception as e:
        #     # print(f"Warning: Could not evaluate rule {rule.rule_id}: {e}")
        return False

    def _draw_rules_panel(self, current_action):
        """Draw right panel showing all rules and their states"""
        panel_x = self.width - self.panel_width
        panel_y = 0

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, self.DARK_GRAY, (panel_x, 0), (panel_x, self.height), 2)

        # Panel title
        title = self.font.render("ACTIVE RULES", True, self.CYAN)
        self.screen.blit(title, (panel_x + 15, 15))

        # Performance metrics - place in top right corner
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            fps_actual = 1000.0 / avg_time if avg_time > 0 else 0
            fps_text = self.tiny_font.render(f"FPS: {fps_actual:.1f}", True, self.GRAY)
            self.screen.blit(fps_text, (panel_x + self.panel_width - 70, 18))

        # Get current observations
        current_obs = self.env.current_observations

        y_offset = 55
        start_y = 55

        # Reset tooltip before checking for new one
        self.tooltip_text = None

        # Clamp scroll offset
        max_scroll = max(0, len(self.env.rules) * 68 - (self.height - 200 - start_y))
        self.rules_scroll_offset = max(0, min(self.rules_scroll_offset, max_scroll))

        # Draw each rule (with scrolling)
        mouse_pos = pygame.mouse.get_pos()
        visible_count = 0
        for idx, rule in enumerate(self.env.rules):
            # Calculate position with scroll offset
            rule_y = y_offset - self.rules_scroll_offset

            # Skip if above visible area
            if rule_y + 68 < start_y:
                y_offset += 68
                continue

            # Stop if below visible area
            if rule_y > self.height - 200:
                # Show "more rules" indicator if there are more rules
                if idx < len(self.env.rules):
                    more_text = self.small_font.render(
                        f"v Scroll for {len(self.env.rules) - idx} more v",
                        True, self.CYAN
                    )
                    text_rect = more_text.get_rect(center=(panel_x + self.panel_width // 2, self.height - 190))
                    self.screen.blit(more_text, text_rect)
                break

            visible_count += 1

            # Check if rule's preconditions are met
            is_active = self._check_rule_condition_met(rule, current_action, current_obs)

            # Rule container
            rule_height = 60
            rule_rect = pygame.Rect(panel_x + 10, rule_y, self.panel_width - 20, rule_height)

            # Highlight if rule is currently active (preconditions met)
            if is_active:
                pygame.draw.rect(self.screen, (50, 80, 60), rule_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.GREEN, rule_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(self.screen, (35, 40, 55), rule_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.DARK_GRAY, rule_rect, 1, border_radius=5)

            # Rule ID
            rule_id_text = self.tiny_font.render(rule.rule_id, True, self.GRAY)
            self.screen.blit(rule_id_text, (panel_x + 18, rule_y + 6))

            # Rule description (wrapped if needed)
            desc = rule.description
            is_truncated = len(desc) > 35
            if is_truncated:
                desc = desc[:32] + "..."

            desc_text = self.small_font.render(desc, True, self.WHITE)
            desc_rect = desc_text.get_rect(topleft=(panel_x + 18, rule_y + 22))
            self.screen.blit(desc_text, desc_rect)

            # Check for tooltip hover
            if is_truncated and desc_rect.collidepoint(mouse_pos):
                self.tooltip_text = rule.description

            # Rule step count
            #step_info = self.tiny_font.render(f"Steps: {rule.step_count}", True, self.LIGHT_GRAY)
            #self.screen.blit(step_info, (panel_x + 18, rule_y + 42))

            # Show active indicator if preconditions met
            if is_active:
                state_indicator = self.tiny_font.render("* ACTIVE", True, self.GREEN)
                self.screen.blit(state_indicator, (panel_x + self.panel_width - 80, rule_y + 42))

            y_offset += 68

        # Draw controls info at bottom
        # Adjust height based on whether auto_policy is available (adds extra line)
        controls_y = self.height - 195 if self.auto_policy else self.height - 175
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 10, controls_y),
                        (panel_x + self.panel_width - 10, controls_y), 1)

        controls_title = self.small_font.render("CONTROLS", True, self.CYAN)
        self.screen.blit(controls_title, (panel_x + 15, controls_y + 10))

        # More compact controls list
        controls = [
            "Keys/Click: Toggle buttons",
            "Scroll: Navigate rules",
            "TAB: Hide/show panel",
            "ENTER: Single step",
            "P: Pause/Resume",
            "+/-: Speed",
            "SPACE: Auto policy" if self.auto_policy else "R: Reset  Q: Quit",
            "R: Reset  Q: Quit" if self.auto_policy else ""
        ]

        cy = controls_y + 35
        for control in controls:
            if control:
                text = self.small_font.render(control, True, self.LIGHT_GRAY)
                self.screen.blit(text, (panel_x + 15, cy))
                cy += 16

    def _draw_tooltip(self, text: str):
        """Draw a tooltip with the given text near the mouse cursor."""
        if not text:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Tooltip appearance
        padding = 8
        max_width = self.panel_width - 30  # Fit within the panel

        # Split text into lines
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            if self.small_font.size(current_line + " " + word)[0] < max_width:
                current_line += " " + word
            else:
                lines.append(current_line.strip())
                current_line = word
        lines.append(current_line.strip())

        # Create text surfaces
        line_surfaces = [self.small_font.render(line, True, self.BLACK) for line in lines]
        tooltip_height = sum(s.get_height() for s in line_surfaces) + padding * 2
        tooltip_width = max(s.get_width() for s in line_surfaces) + padding * 2

        # Position tooltip
        tooltip_x = mouse_x + 15
        tooltip_y = mouse_y + 15

        # Adjust if it goes off-screen
        if tooltip_x + tooltip_width > self.width:
            tooltip_x = mouse_x - tooltip_width - 15
        if tooltip_y + tooltip_height > self.height:
            tooltip_y = mouse_y - tooltip_height - 15

        # Draw tooltip background
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, self.YELLOW, tooltip_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.BLACK, tooltip_rect, 1, border_radius=5)

        # Draw text lines
        current_y = tooltip_y + padding
        for surface in line_surfaces:
            self.screen.blit(surface, (tooltip_x + padding, current_y))
            current_y += surface.get_height()

    def _handle_mouse_click(self, pos: tuple, button_positions: List[tuple]):
        """Handle mouse click on buttons"""
        x, y = pos

        for i, (bx, by) in enumerate(button_positions):
            distance = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
            if distance <= self.button_radius:
                return i

        return None

    def _get_action_from_buttons(self) -> torch.Tensor:
        """Convert button states to action tensor"""
        action = torch.zeros(self.env.action_dim)
        for i, pressed in enumerate(self.button_pressed):
            action[i] = 1.0 if pressed else 0.0
        return action

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if the user quits."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False

                elif event.key == pygame.K_r:
                    self.env.reset()
                    self.button_pressed = [False] * self.env.action_dim
                    print("Environment reset")

                elif event.key == pygame.K_SPACE and self.auto_policy:
                    self.auto_mode = not self.auto_mode
                    print(f"Auto policy: {'ON' if self.auto_mode else 'OFF'}")

                elif event.key == pygame.K_TAB:
                    self.show_rules_panel = not self.show_rules_panel
                    print(f"Rules panel: {'SHOWN' if self.show_rules_panel else 'HIDDEN'}")

                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Stepping: {'PAUSED' if self.paused else 'RUNNING'}")

                elif event.key == pygame.K_RETURN:
                    if self.paused:
                        self.steps_since_last_update = 1
                        print("Single step")

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.auto_step_speed = min(240, self.auto_step_speed + 10)
                    print(f"Speed: {self.auto_step_speed} steps/sec")

                elif event.key == pygame.K_MINUS:
                    self.auto_step_speed = max(1, self.auto_step_speed - 10)
                    print(f"Speed: {self.auto_step_speed} steps/sec")

                elif event.unicode in self.key_to_button:
                    button_idx = self.key_to_button[event.unicode]
                    if button_idx < self.env.action_dim:
                        self.button_pressed[button_idx] = True

            elif event.type == pygame.KEYUP:
                if event.unicode in self.key_to_button:
                    button_idx = self.key_to_button[event.unicode]
                    if button_idx < self.env.action_dim:
                        self.button_pressed[button_idx] = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                button_positions = self._get_button_positions()
                if event.button == 1:
                    button_idx = self._handle_mouse_click(event.pos, button_positions)
                    if button_idx is not None:
                        if button_idx in self.mouse_held_buttons:
                            self.mouse_held_buttons.remove(button_idx)
                            self.button_pressed[button_idx] = False
                        else:
                            self.mouse_held_buttons.add(button_idx)
                            self.button_pressed[button_idx] = True
                elif event.button == 4:
                    if self.show_rules_panel:
                        self.rules_scroll_offset -= 40
                elif event.button == 5:
                    if self.show_rules_panel:
                        self.rules_scroll_offset += 40

        return True

    def close(self):
        """Close the pygame interface."""
        pygame.quit()
    def render(self, obs: torch.Tensor, action: torch.Tensor):
        """Render the environment state."""
        self.screen.fill(self.BG_DARK)
        self._draw_header()

        if self.show_rules_panel:
            self._draw_rules_panel(action)

        button_positions = self._get_button_positions()
        for i, pos in enumerate(button_positions):
            self._draw_button(pos, i, action[i] > 0.5)

        slot_positions = self._get_slot_positions()
        for i, pos in enumerate(slot_positions):
            self._draw_slot(pos, i, obs[i].item())

        if self.tooltip_text:
            self._draw_tooltip(self.tooltip_text)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        """Main game loop for interactive use, with controlled stepping speed."""
        obs = self.env.reset()
        running = True

        # Initialize / reset time accumulator for smooth stepping
        if not hasattr(self, 'time_accumulator'):
            self.time_accumulator = 0.0

        while running:
            # --- 1. Frame Timing and Event Handling ---
            frame_start = time.time()
            # dt is the time elapsed since the last frame based on self.fps
            dt = 1.0 / self.fps 

            # self.handle_events() should update self.paused, 
            # self.steps_since_last_update, and the button states.
            running = self.handle_events()
            
            # Determine how many steps to take this frame
            num_steps_this_frame = 0

            # --- 2. Calculate Steps Based on Time/Speed (The Core Logic) ---
            if not self.paused:
                # Auto-stepping based on speed (self.auto_step_speed steps/sec)
                self.time_accumulator += dt
                steps_per_second = self.auto_step_speed
                time_per_step = 1.0 / steps_per_second

                # Calculate how many full steps have accumulated in time
                while self.time_accumulator >= time_per_step:
                    self.time_accumulator -= time_per_step
                    num_steps_this_frame += 1

            # --- 3. Single-Step Override (When paused and 'Enter' is pressed) ---
            # The 'handle_events' method must set this flag.
            if self.steps_since_last_update > 0:
                num_steps_this_frame += self.steps_since_last_update
                self.steps_since_last_update = 0

            # --- 4. Step Environment (Action Determined Inside Loop) ---
            action = self._get_action_from_buttons()

            if num_steps_this_frame > 0:
                for _ in range(num_steps_this_frame):
                    # Determine action based on mode for THIS step
                    if self.auto_mode and self.auto_policy:
                        # Use auto policy's action
                        action = self.auto_policy(obs)
                        # Update button display to match auto actions (for rendering)
                        for i in range(self.env.action_dim):
                            self.button_pressed[i] = action[i] > 0.5

                    # Step the environment
                    obs, info = self.env.step(lambda _obs: action)
                    obs = obs[-1] # use last observation
            self.render(obs, action)

            # --- 6. Frame Rate Control ---
            # This limits the *display* FPS, not the *environment step* FPS.
            self.clock.tick(self.fps)

        pygame.quit()
        print("Interface closed")
# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os

    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from environments.switchboard.switchboard import Switchboard

    parser = argparse.ArgumentParser(
        description='Run Switchboard scenario with Pygame interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load a scenario
  python pygame_interface.py --scenario direct_rules

  # Load a scenario with custom window size
  python pygame_interface.py --scenario challenging_rules --width 1400 --height 800

  # Create empty switchboard for manual rule testing
  python pygame_interface.py --action_dim 5 --obs_dim 5
        """
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='direct_rules',
        help='Name of the scenario to load (e.g., direct_rules, challenging_rules)'
    )

    parser.add_argument(
        '--action_dim',
        type=int,
        default=10,
        help='Number of action buttons (default: 10)'
    )

    parser.add_argument(
        '--obs_dim',
        type=int,
        default=10,
        help='Number of observation slots (default: 10)'
    )


    parser.add_argument(
        '--width',
        type=int,
        default=1200,
        help='Window width (default: 1200)'
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
    env = Switchboard(
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        time_scaling=0 # disable time scaling
    )

    print("=" * 60)
    print("SWITCHBOARD PYGAME INTERFACE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Action dimension: {args.action_dim}")
    print(f"  Observation dimension: {args.obs_dim}")
    print(f"  Window size: {args.width}x{args.height}")

    # Load rules from scenario if provided
    if args.scenario:
        print(f"\nLoading scenario: {args.scenario}")
        try:
            env.load_rules(args.scenario)
        except (ValueError, ImportError) as e:
            print(f"\nX ERROR: {e}")
            sys.exit(1)

        print(f"\nTotal rules: {len(env.rules)}")
        print("Explore the scenario - can you discover all the patterns?")
    else:
        print("\nNo rules loaded - empty switchboard")
        print("Use this to manually test rule creation")

    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("  Buttons:")
    print(f"    - 0-9 Keys: Press buttons 0-{min(9, args.action_dim-1)}")
    print("    - Mouse Click: Toggle buttons on/off")
    print("  Navigation:")
    print("    - Scroll: Navigate rules panel")
    print("    - TAB: Hide/show rules panel")
    print("  Stepping:")
    print("    - P: Pause/Resume auto-stepping")
    print("    - ENTER: Single step (when paused)")
    print("    - +/-: Adjust speed")
    print("  Other:")
    print("    - R: Reset environment")
    print("    - Q/ESC: Quit")
    print("\nStarting PAUSED for manual exploration")
    print("=" * 60)
    print()

    # Create pygame interface
    interface = SwitchboardPygameInterface(
        env,
        width=args.width,
        height=args.height,
        fps=args.fps,
        auto_policy=None
    )

    # Run the interface
    interface.run()
