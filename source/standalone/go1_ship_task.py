import math
import torch

from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.assets import load_articulation_cfg


class Go1ShipEnv(DirectRLEnv):
    def __init__(self, sim_device, graphics_device, headless):
        super().__init__(sim_device, graphics_device, headless)

        # === Load robot (Unitree Go1) ===
        go1_cfg = load_articulation_cfg("UnitreeGo1")
        self.robot = Articulation(go1_cfg)

        # === Load platform (simple flat box as ship deck) ===
        platform_cfg = {
            "prim_path": "/World/Platform",
            "usd_path": "${ISAACLAB_PATH}/usd/props/blocks/cube.usd",
            "scale": [5.0, 5.0, 0.1],
            "init_state": {"pos": [0.0, 0.0, 0.0]},
            "rigid_props": {"disable_gravity": True},
        }
        self.platform = Articulation(platform_cfg)

        # === Goal position (2m forward) ===
        self.goal_pos = torch.tensor([2.0, 0.0], device=sim_device)

        # Time counter for oscillation
        self.t = 0.0

    def set_up_scene(self):
        self.scene.add(self.platform)
        self.scene.add(self.robot)

    def pre_physics_step(self, actions):
        # Apply robot actions (motor torques or positions)
        self.robot.apply_action(actions)

    def post_physics_step(self, dt: float):
        self.t += dt
        self._oscillate_platform(self.t)

        obs = self.compute_observations()
        rew = self.compute_rewards(obs)
        done = self.check_termination(obs)
        return obs, rew, done, {}

    def compute_observations(self):
        robot_pos, _ = self.robot.get_world_poses()
        obs = {
            "proprio": self.robot.get_proprioception(),
            "goal": self.goal_pos - robot_pos[..., :2],  # relative xy distance
        }
        return obs

    def compute_rewards(self, obs):
        robot_pos, _ = self.robot.get_world_poses()
        dist_to_goal = torch.norm(self.goal_pos - robot_pos[..., :2], dim=-1)
        reward = -dist_to_goal
        return reward

    def check_termination(self, obs):
        robot_pos, _ = self.robot.get_world_poses()
        return (robot_pos[..., 2] < 0.2)

    def _oscillate_platform(self, t):
        """Make the platform oscillate like a ship on waves"""
        amplitude = 0.2  # radians
        frequency = 0.5  # Hz
        angle = amplitude * math.sin(2 * math.pi * frequency * t)
        self.platform.set_joint_positions(torch.tensor([angle], device=self.sim_device))


if __name__ == "__main__":
    env = Go1ShipEnv(sim_device="cuda:0", graphics_device="cuda:0", headless=False)
    obs = env.reset()
    for i in range(1000):
        # Dummy random actions (replace with RL policy later)
        actions = torch.randn((env.robot.num_actions,), device=env.sim_device)
        obs, rew, done, _ = env.step(actions)
        if done:
            obs = env.reset()
