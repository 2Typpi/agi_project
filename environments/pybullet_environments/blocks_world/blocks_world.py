import pybullet as p
import pybullet_data
import numpy as np
import time
import torch

from typing import Callable, Dict, List
from abc import ABC, abstractmethod

from environments.realtime_environment import RealTimeEnvironment


class BlocksWorld(RealTimeEnvironment):
    """
    PyBullet-based 3D world environment with egocentric vision.
    Agent is a controllable box in a room with various objects.
    """
    
    def __init__(
        self,
        gui: bool = True,
        image_width: int = 128,
        image_height: int = 128,
        dt: float = 1/240,
        reset_agent: bool = True,
        reset_objects: bool = False,
        randomize_agent_position: bool = True,
        randomize_agent_orientation: bool = True,
        randomize_object_positions: bool = False,
        **kwargs
    ):
        """
        Args:
            gui: Whether to show PyBullet GUI
            image_width: Width of rendered images
            image_height: Height of rendered images
            dt: Physics timestep
            reset_agent: Whether to reset agent on environment reset
            reset_objects: Whether to reset objects to initial positions on reset
            randomize_agent_position: Whether to randomize agent position on reset
            randomize_agent_orientation: Whether to randomize agent orientation on reset
            randomize_object_positions: Whether to randomize object positions on reset
            **kwargs: Arguments passed to RealTimeEnvironment
        """
        self.gui = gui
        self.image_width = image_width
        self.image_height = image_height

        # Reset behavior settings
        self.reset_agent = reset_agent
        self.reset_objects = reset_objects
        self.randomize_agent_position = randomize_agent_position
        self.randomize_agent_orientation = randomize_agent_orientation
        self.randomize_object_positions = randomize_object_positions

        self.physics_client = None
        self.agent_id = None
        self.room_objects = []
        self.world_objects = []

        # Store initial object states
        self.initial_object_states = []

        # Physics parameters
        self.dt = dt

        # Camera parameters
        self.camera_distance = 0.0  # Camera at agent center (prevents wall clipping)
        self.camera_height = 0.2    # Camera height above agent center

        # Setup physics before calling super().__init__
        self._setup_physics()
        super().__init__(**kwargs)
    
    def _setup_physics(self):
        """Initialize PyBullet physics simulation"""
        # Connect to PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Create the world
        self._create_room()
        self._create_agent()
        self._create_objects()
        
        # Set up lighting for better rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0 if not self.gui else 1)

    def _create_room(self):
        """Create a walled room with skybox"""
        room_size = 10.0
        wall_height = 3.0
        wall_thickness = 0.2
        
        # Floor
        floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[room_size/2, room_size/2, 0.1])
        floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[room_size/2, room_size/2, 0.1], 
                                          rgbaColor=[0.8, 0.8, 0.8, 1])
        floor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision,
                                   baseVisualShapeIndex=floor_visual, basePosition=[0, 0, -0.1])
        self.room_objects.append(floor_id)
        
        # Walls
        wall_positions = [
            [room_size/2, 0, wall_height/2],   # Right wall
            [-room_size/2, 0, wall_height/2],  # Left wall  
            [0, room_size/2, wall_height/2],   # Back wall
            [0, -room_size/2, wall_height/2]   # Front wall
        ]
        
        wall_orientations = [
            [0, 0, 0, 1],  # Right
            [0, 0, 0, 1],  # Left
            [0, 0, 0, 1],  # Back
            [0, 0, 0, 1]   # Front
        ]
        
        wall_half_extents = [
            [wall_thickness/2, room_size/2, wall_height/2],  # Right
            [wall_thickness/2, room_size/2, wall_height/2],  # Left
            [room_size/2, wall_thickness/2, wall_height/2],  # Back
            [room_size/2, wall_thickness/2, wall_height/2]   # Front
        ]
        
        for i, (pos, orn, extents) in enumerate(zip(wall_positions, wall_orientations, wall_half_extents)):
            wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, 
                                            rgbaColor=[0.7, 0.7, 0.9, 1])
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                                      baseVisualShapeIndex=wall_visual, basePosition=pos,
                                      baseOrientation=orn)
            self.room_objects.append(wall_id)
        
        # Ceiling (optional skybox effect)
        ceiling_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[room_size/2, room_size/2, 0.1])
        ceiling_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[room_size/2, room_size/2, 0.1], 
                                           rgbaColor=[0.6, 0.8, 1.0, 1])
        ceiling_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ceiling_collision,
                                     baseVisualShapeIndex=ceiling_visual, 
                                     basePosition=[0, 0, wall_height + 0.1])
        self.room_objects.append(ceiling_id)
    
    def _create_agent(self):
        """Create the controllable agent (box)"""
        # Agent is a box
        agent_size = [0.2, 0.2, 0.2]  # width, depth, height
        agent_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=agent_size)
        agent_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=agent_size, 
                                         rgbaColor=[1.0, 0.2, 0.2, 1])
        
        self.agent_id = p.createMultiBody(baseMass=1.0, 
                                        baseCollisionShapeIndex=agent_collision,
                                        baseVisualShapeIndex=agent_visual,
                                        basePosition=[0, 0, agent_size[2] + 0.25])
        
        # Set agent dynamics - reduce friction significantly
        p.changeDynamics(self.agent_id, -1, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0)
    
    def _create_objects(self):
        """Create various objects in the world"""
        # Create different types of objects
        object_configs = [
            # Spheres
            {'type': 'sphere', 'pos': [2, 2, 0.5], 'radius': 0.3, 'color': [0.2, 0.8, 0.2, 1], 'mass': 0.5},
            {'type': 'sphere', 'pos': [-2, 2, 0.4], 'radius': 0.25, 'color': [0.8, 0.8, 0.2, 1], 'mass': 0.3},
            {'type': 'sphere', 'pos': [3, -1, 0.3], 'radius': 0.2, 'color': [0.8, 0.2, 0.8, 1], 'mass': 0.2},
            
            # Boxes
            {'type': 'box', 'pos': [-3, -2, 0.5], 'size': [0.4, 0.4, 0.8], 'color': [0.2, 0.2, 0.8, 1], 'mass': 1.0},
            {'type': 'box', 'pos': [1, -3, 0.3], 'size': [0.6, 0.3, 0.4], 'color': [0.8, 0.4, 0.2, 1], 'mass': 0.8},
            {'type': 'box', 'pos': [-1, 3, 0.4], 'size': [0.3, 0.5, 0.6], 'color': [0.4, 0.8, 0.4, 1], 'mass': 0.6},
            
            # Cylinders
            {'type': 'cylinder', 'pos': [3, 1, 0.5], 'radius': 0.2, 'height': 0.8, 'color': [0.6, 0.2, 0.8, 1], 'mass': 0.7},
            {'type': 'cylinder', 'pos': [-2, -1, 0.4], 'radius': 0.15, 'height': 0.6, 'color': [0.8, 0.6, 0.2, 1], 'mass': 0.4},
        ]
        
        for config in object_configs:
            if config['type'] == 'sphere':
                collision = p.createCollisionShape(p.GEOM_SPHERE, radius=config['radius'])
                visual = p.createVisualShape(p.GEOM_SPHERE, radius=config['radius'], 
                                           rgbaColor=config['color'])
                
            elif config['type'] == 'box':
                half_extents = [s/2 for s in config['size']]
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                           rgbaColor=config['color'])
                
            elif config['type'] == 'cylinder':
                collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=config['radius'], 
                                                 height=config['height'])
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=config['radius'], 
                                           length=config['height'], rgbaColor=config['color'])
            
            obj_id = p.createMultiBody(baseMass=config['mass'],
                                     baseCollisionShapeIndex=collision,
                                     baseVisualShapeIndex=visual,
                                     basePosition=config['pos'])
            self.world_objects.append(obj_id)

            # Store initial state for resetting
            self.initial_object_states.append({
                'id': obj_id,
                'position': config['pos'],
                'orientation': [0, 0, 0, 1],  # Default quaternion
                'config': config
            })
    
    def _get_initial_state(self) -> torch.Tensor:
        """Get initial state (egocentric image)"""
        # Reset agent if configured to do so
        if self.reset_agent:
            if self.randomize_agent_position:
                x = (torch.rand(1, device=self.device) - 0.5) * 6  # Random x in [-3, 3]
                y = (torch.rand(1, device=self.device) - 0.5) * 6  # Random y in [-3, 3]
            else:
                x, y = 0.0, 0.0

            if self.randomize_agent_orientation:
                angle = torch.rand(1, device=self.device) * 2 * torch.pi  # Random orientation
            else:
                angle = 0.0

            # Reset agent position and orientation
            agent_pos = [x if isinstance(x, float) else x.item(),
                        y if isinstance(y, float) else y.item(),
                        0.3]
            agent_orn = p.getQuaternionFromEuler([0, 0, angle if isinstance(angle, float) else angle.item()])
            p.resetBasePositionAndOrientation(self.agent_id, agent_pos, agent_orn)
            p.resetBaseVelocity(self.agent_id, [0, 0, 0], [0, 0, 0])

        # Reset objects if configured to do so
        if self.reset_objects:
            for obj_state in self.initial_object_states:
                if self.randomize_object_positions:
                    # Randomize position within room bounds
                    x = (torch.rand(1, device=self.device) - 0.5) * 6
                    y = (torch.rand(1, device=self.device) - 0.5) * 6
                    z = obj_state['position'][2]  # Keep original height
                    pos = [x.item(), y.item(), z]
                    angle = torch.rand(1, device=self.device) * 2 * torch.pi
                    orn = p.getQuaternionFromEuler([0, 0, angle.item()])
                else:
                    pos = obj_state['position']
                    orn = obj_state['orientation']

                p.resetBasePositionAndOrientation(obj_state['id'], pos, orn)
                p.resetBaseVelocity(obj_state['id'], [0, 0, 0], [0, 0, 0])

        return self._get_observation()
    
    def _get_observation(self) -> torch.Tensor:
        """Get egocentric image observation"""
        # Get agent position and orientation
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.agent_id)
        agent_euler = p.getEulerFromQuaternion(agent_orn)
        
        # Calculate camera position (slightly above and in front of agent)
        camera_x = agent_pos[0] + self.camera_distance * np.cos(agent_euler[2])
        camera_y = agent_pos[1] + self.camera_distance * np.sin(agent_euler[2])
        camera_z = agent_pos[2] + self.camera_height
        
        # Calculate target position (where camera is looking)
        target_distance = 2.0
        target_x = camera_x + target_distance * np.cos(agent_euler[2])
        target_y = camera_y + target_distance * np.sin(agent_euler[2])
        target_z = camera_z
        
        # Create view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[camera_x, camera_y, camera_z],
            cameraTargetPosition=[target_x, target_y, target_z],
            cameraUpVector=[0, 0, 1]
        )
        
        # Create projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.image_width / self.image_height,
            nearVal=0.01,  # Reduced near plane to prevent clipping
            farVal=100.0
        )
        
        # Render image
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to tensor (RGB image)
        rgb_array = np.array(rgb_img[:, :, :3])  # Remove alpha channel
        rgb_tensor = torch.from_numpy(rgb_array).float().to(self.device)
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # Change to CHW format
        rgb_tensor = rgb_tensor / 255.0  # Normalize to [0, 1]
        
        return rgb_tensor
    def _get_state(self):
        # get position, velocities etc. of all objects:
        state = {}
        # Agent
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)
        state['agent'] = {
            'position': agent_pos,
            'orientation': agent_orn,
            'velocity': agent_vel,
            'angular_velocity': agent_ang_vel
        }
        # Room objects (static)
        state['room'] = []
        for obj_id in self.room_objects:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            state['room'].append({
                'id': obj_id,
                'position': pos,
                'orientation': orn
            })
        # World objects (dynamic)
        state['world'] = []
        for obj_id in self.world_objects:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            vel, ang_vel = p.getBaseVelocity(obj_id)
            state['world'].append({
                'id': obj_id,
                'position': pos,
                'orientation': orn,
                'velocity': vel,
                'angular_velocity': ang_vel
            })
        return state
    
    def _step_simulation(self, action: torch.Tensor) -> torch.Tensor:
        """
        Perform one simulation step
        
        Args:
            action: [forward_velocity, rotation_velocity] continuous actions
            
        Returns:
            observation: Egocentric image after simulation step
        """
        # Parse action
        if action.numel() >= 2:
            forward_vel = 10*torch.clamp(action[0], -1.0, 1.0).item()  # Forward/backward
            rotation_vel = 100*torch.clamp(action[1], -1.0, 1.0).item()  # Rotation
        else:
            forward_vel = 0.0
            rotation_vel = 0.0
        
        # Get current agent state
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.agent_id)
        agent_euler = p.getEulerFromQuaternion(agent_orn)
        
        # Calculate movement in world coordinates
        forward_x = forward_vel * np.cos(agent_euler[2]) * self.dt
        forward_y = forward_vel * np.sin(agent_euler[2]) * self.dt
        
        # Apply velocities
        current_vel, current_ang_vel = p.getBaseVelocity(self.agent_id)
        # Clamp vertical velocity to prevent wall climbing (only allow falling, not rising)
        vertical_vel = min(current_vel[2], 0.0)
        new_vel = [forward_x / self.dt, forward_y / self.dt, vertical_vel]
        # Keep agent level (only allow yaw rotation, no pitch/roll)
        new_ang_vel = [0.0, 0.0, rotation_vel]

        p.resetBaseVelocity(self.agent_id, new_vel, new_ang_vel)

        # Step physics
        p.stepSimulation()

        # Keep agent upright by resetting orientation
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.agent_id)
        agent_euler = p.getEulerFromQuaternion(agent_orn)
        # Preserve only yaw (z-axis rotation), reset pitch and roll to 0
        upright_orn = p.getQuaternionFromEuler([0, 0, agent_euler[2]])
        p.resetBasePositionAndOrientation(self.agent_id, agent_pos, upright_orn)
        
        # Update step count
        self.step_count += 1
        
        # Get new observation
        self.state = self._get_observation()
        return self.state.clone()
    
    def render(self):
        """Render is handled automatically by PyBullet GUI"""
        pass
    
    def close(self):
        """Clean up PyBullet resources"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            self.agent_id = None
            self.room_objects = []
            self.world_objects = []
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()