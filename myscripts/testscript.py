from isaaclab.app import AppLauncher
import gymnasium as gym
import torch
import imageio
import numpy as np
import torch.nn as nn

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

LR = 3e-4
DEVICE = "cuda"
TOTAL_TIMESTEPS = 1e6
TIMESTEPS = 200
NUM_ENVS = 4

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Agent, self).__init__()
        self.actor_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic_net(x)

    def get_action_and_value(self, x, action=None):
        # normal distribution for continuous action spaces
        action_mean = self.actor_net(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)




class MyIsaacEnv:
    def __init__(self, env_name):

        # load environment configuration
        # create base environment
        self.cfg = load_cfg_from_registry(env_name, "env_cfg_entry_point")
        self.cfg.scene.num_envs = NUM_ENVS  # set number of parallel environments to 1 for testing
        # self.add_camera_view()

        self.env = gym.make(env_name, cfg=self.cfg, render_mode="rgb_array")
        # wrap environment to enforce that reset is called before step
        self.env = gym.wrappers.OrderEnforcing(self.env)
        self.env.reset()

        self.obs_dim = self.env.observation_space["policy"].shape[1]
        self.action_dim = self.env.action_space.shape[1]
        print(f"Observation space Shape: {self.obs_dim}")
        print(f"Action space Shape: {self.action_dim}")

        # initializa agent
        self.agent = Agent(self.obs_dim, self.action_dim).to(DEVICE)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=LR)


    def add_camera_view(self):  # optional
        # adjust camera resolution and pose
        self.cfg.viewer.resolution = (720, 720)
        self.cfg.viewer.eye = (3.0, 3.0, 3.0)
        self.cfg.viewer.lookat = (0.0, 1.0, 2.0)

    def TakeRandomActions(self):
        frames = []
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(torch.tensor(action))
            frame = self.env.render()
            frames.append(frame)

        imageio.mimwrite("test_video.mp4", frames, fps=30)
        print("Saved video to test_video.mp4")
        # ✅ clean shutdown
        self.env.close()

    def TrainAgent(self):
        updates = int(TOTAL_TIMESTEPS // TIMESTEPS)
        done = torch.zeros(NUM_ENVS).to(DEVICE)

        # storage
        self.obs_arr = torch.zeros((TIMESTEPS, NUM_ENVS, self.obs_dim)).to(DEVICE)
        self.actions_arr = torch.zeros((TIMESTEPS, NUM_ENVS, self.action_dim)).to(DEVICE)
        self.logprobs_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.rewards_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.dones_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.values_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)

        for update in range(updates):
            obs, _ = self.env.reset()
            for step in range(TIMESTEPS):
                obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(obs)

                self.obs_arr[step] = obs
                self.actions_arr[step] = action
                self.dones_arr[step] = done
                self.values_arr[step] = value.flatten()
                self.logprobs_arr[step] = log_prob

                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = torch.tensor(terminated | truncated).to(DEVICE)
                self.rewards_arr[step] = torch.from_numpy(reward).float().to(DEVICE)
            

            # Calculate returns and Advantage

            # prep data structures for PPO update

            # PPO update

                
    def TestAgent(self):
        pass


if __name__ == "__main__":
    myenv = MyIsaacEnv("Isaac-Cartpole-v0")
    myenv.TakeRandomActions()
    simulation_app.close()