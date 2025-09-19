import math
import torch
import gym
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.utils.math import quat_from_euler_xyz
from omni.isaac.lab_assets import UNITREE_GO1_CFG


class Go1Env(DirectRLEnv):
    """Direct-based locomotion task for Unitree Go1 on a ship-like platform."""

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        super().__init__(cfg, sim_device, graphics_device_id, headless)

        # Action space: 12 DOF joints
        self.num_actions = 12
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=float
        )

        # PD gains
        self.kp = 40.0
        self.kd = 1.0

        # Default standing pose
        self.default_joint_pos = torch.tensor([
            0.0, 0.9, -1.8,   # front-left
            0.0, 0.9, -1.8,   # front-right
            0.0, 0.9, -1.8,   # rear-left
            0.0, 0.9, -1.8    # rear-right
        ], device=self.sim_device)

        # Goal position (20m ahead)
        self.goal_pos = torch.tensor([20.0, 0.0, 0.0], device=self.sim_device)

        # Ship motion params
        self.wave_amp = 0.1
        self.wave_freq = 0.5

        # Time
        self.time = 0.0
        self.dt = 1.0 / 60.0

    def set_up_scene(self, scene):
        """Create robot and platform."""
        self.robot = Articulation(UNITREE_GO1_CFG.replace(prim_path="/World/Go1"))
        scene.add(self.robot)

        self.platform = RigidObject(
            prim_path="/World/platform",
            name="platform",
            usd_path=None,
            shape="cube",
            scale=(20.0, 20.0, 0.5),
        )
        scene.add(self.platform)

    def pre_physics_step(self, actions):
        """Apply actions and update platform transform."""
        self.time += self.dt

        actions = torch.clamp(torch.tensor(actions, device=self.sim_device), -1.0, 1.0)
        target_joint_pos = self.default_joint_pos + 0.5 * actions
        self.robot.set_joint_position_targets(target_joint_pos, kp=self.kp, kd=self.kd)

        # Ship-like sinusoidal motion
        roll = self.wave_amp * math.sin(2 * math.pi * self.wave_freq * self.time)
        pitch = self.wave_amp * math.sin(2 * math.pi * 0.8 * self.wave_freq * self.time)
        roll += 0.02 * torch.randn(1).item()
        pitch += 0.02 * torch.randn(1).item()

        q = quat_from_euler_xyz(roll, pitch, 0.0)
        self.platform.set_world_pose(position=(0, 0, 0), orientation=q)

    def get_observations(self):
        """Joint states, base velocities, and goal vector."""
        obs = torch.cat([
            self.robot.get_joint_positions(),
            self.robot.get_joint_velocities(),
            self.robot.data.root_state[:, 7:10],   # linear vel
            self.robot.data.root_state[:, 10:13],  # angular vel
            self.goal_pos - self.robot.data.root_state[:, 0:3],
        ], dim=-1)
        return {"policy": obs}

    def calculate_metrics(self):
        """Reward for moving toward goal and staying upright."""
        root_pos = self.robot.data.root_state[:, 0:3]
        dist_to_goal = torch.norm(self.goal_pos - root_pos, dim=-1)

        reward_goal = -dist_to_goal
        reward_alive = 1.0
        reward_upright = torch.exp(-torch.abs(self.robot.data.root_state[:, 5]))

        rew = reward_goal + 0.5 * reward_alive + 0.1 * reward_upright
        return {"reward": rew}

    def is_done(self):
        """Terminate when fallen or reached goal."""
        root_pos = self.robot.data.root_state[:, 0:3]
        z = root_pos[:, 2]
        done = (z < 0.2) | (torch.norm(self.goal_pos - root_pos, dim=-1) < 0.5)
        return done
