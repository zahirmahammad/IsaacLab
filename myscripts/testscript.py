from isaaclab.app import AppLauncher
import gymnasium as gym
import torch
import imageio
import numpy as np
import torch.nn as nn
import cv2

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True, enable_cameras=True)
# app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

LR = 5e-5  # Slightly increased learning rate
DEVICE = "cuda"
TOTAL_TIMESTEPS = 2e7
TIMESTEPS = 256
NUM_ENVS = 256
GAMMA = 0.99
LAMBDA = 0.95
BATCH_SIZE = 1024
NUM_EPOCHS = 5
ENT_COEF = 0.01  # Reduced entropy coefficient to focus on task
CLIP_RATIO = 0.2
MAX_GRAD_NORM = 0.5
JUMPING_PENALTY = 0.5  # Strong penalty on action magnitude - discourages explosive movements


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Agent, self).__init__()
        # Increase network capacity and add normalization layers
        self.actor_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 1), std=1.0),
        )
        
        # Running statistics for observation normalization
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def update_obs_norm(self, obs):
        """Update running statistics for observation normalization"""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]
        
        # Update running mean and variance (Welford's online algorithm)
        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count
        
        self.obs_mean = self.obs_mean + delta * batch_count / tot_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.obs_count * batch_count / tot_count
        self.obs_var = M2 / tot_count
        self.obs_count = tot_count
    
    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

    def get_value(self, x):
        x = self.normalize_obs(x)
        return self.critic_net(x)

    def get_action_and_value(self, x, action=None):
        # Normalize observations
        x_norm = self.normalize_obs(x)
        
        # normal distribution for continuous action spaces
        action_mean = self.actor_net(x_norm)
        action_logstd = torch.clamp(self.actor_logstd, -5, 2)
        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        # action = torch.tanh(action)  # ensure actions are in [-1, 1]
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)




class MyIsaacEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        # load environment configuration
        # create base environment
        self.cfg = load_cfg_from_registry(env_name, "env_cfg_entry_point")
        self.cfg.scene.num_envs = NUM_ENVS  # set number of parallel environments to 1 for testing
        # self.add_camera_view()
        self.env = self.make_env(self.cfg, eval=False)

        self.eval_cfg = load_cfg_from_registry(env_name, "env_cfg_entry_point")
        self.eval_cfg.scene.num_envs = 1  # set number of parallel environments to 1 for testing


        self.obs_dim = self.env.observation_space["policy"].shape[1]
        self.action_dim = self.env.action_space.shape[1]
        print(f"Observation space Shape: {self.obs_dim}")
        print(f"Action space Shape: {self.action_dim}")

        # initializa agent
        self.agent = Agent(self.obs_dim, self.action_dim).to(DEVICE)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=LR)
        
        # Track previous actions for smoothing penalty
        self.prev_actions = None
        


    def make_env(self, cfg, eval=False):
        # if eval:
        env = gym.make(self.env_name, cfg=cfg, render_mode="rgb_array")
        # else:
            # env = gym.make(self.env_name, cfg=cfg)
        env = gym.wrappers.OrderEnforcing(env)
        return env

    def add_camera_view(self):  # optional
        # adjust camera resolution and pose
        self.cfg.viewer.resolution = (720, 720)
        self.cfg.viewer.eye = (3.0, 3.0, 3.0)
        self.cfg.viewer.lookat = (0.0, 1.0, 2.0)

    def TakeRandomActions(self):
        frames = []
        for _ in range(1000):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(torch.tensor(action))
            frame = self.env.render()
            frames.append(frame)

        imageio.mimwrite("test_video.mp4", frames, fps=30)
        print("Saved video to test_video.mp4")
        # ✅ clean shutdown
        self.env.close()

    def TrainAgent(self):
        runs = int(TOTAL_TIMESTEPS // (TIMESTEPS * NUM_ENVS))
        TEST_FREQ = max(1, int(runs // 4))
        done = torch.zeros(NUM_ENVS).to(DEVICE)

        # storage
        self.obs_arr = torch.zeros((TIMESTEPS, NUM_ENVS, self.obs_dim)).to(DEVICE)
        self.actions_arr = torch.zeros((TIMESTEPS, NUM_ENVS, self.action_dim)).to(DEVICE)
        self.logprobs_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.rewards_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.dones_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)
        self.values_arr = torch.zeros((TIMESTEPS, NUM_ENVS)).to(DEVICE)

        for run in range(runs):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            self.prev_actions = torch.zeros((NUM_ENVS, self.action_dim)).to(DEVICE)
            
            for step in range(TIMESTEPS):
                # obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(obs)

                self.obs_arr[step] = obs
                self.actions_arr[step] = action
                self.dones_arr[step] = done
                self.values_arr[step] = value.flatten()
                self.logprobs_arr[step] = log_prob

                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = obs["policy"]
                done = (terminated | truncated).float().to(DEVICE)
                self.rewards_arr[step] = reward
                self.prev_actions = action.clone()
            
            # Update observation normalization statistics
            self.agent.update_obs_norm(self.obs_arr.reshape(-1, self.obs_dim))

            # Calculate returns and Advantage
            advantages = torch.zeros_like(self.rewards_arr).to(DEVICE)
            lastgaelam = torch.zeros(NUM_ENVS).to(DEVICE)
            with torch.no_grad():
                next_value = self.agent.get_value(obs).flatten()
            for r in reversed(range(TIMESTEPS)):
                if r == TIMESTEPS - 1:
                    next_done = 1 - done
                    next_value = next_value
                else:
                    next_done = 1 - self.dones_arr[r + 1]
                    next_value = self.values_arr[r + 1]
                delta = self.rewards_arr[r] + GAMMA * next_value * next_done - self.values_arr[r]
                lastgaelam = delta + GAMMA * LAMBDA * next_done * lastgaelam
                advantages[r] = lastgaelam
            returns = advantages + self.values_arr

            # prep data structures for PPO update
            self.d_obs = self.obs_arr.reshape(-1, self.obs_dim).to(DEVICE)
            self.d_actions = self.actions_arr.reshape(-1, self.action_dim).to(DEVICE)
            self.d_logprobs = self.logprobs_arr.reshape(-1).to(DEVICE)
            self.d_advantages = advantages.reshape(-1).to(DEVICE)
            self.d_returns = returns.reshape(-1).to(DEVICE)
            advantages_normal = (self.d_advantages - self.d_advantages.mean()) / (self.d_advantages.std() + 1e-8)
            
            # PPO update
            data_size = NUM_ENVS * TIMESTEPS
            
            for i in range(NUM_EPOCHS):
                indices = np.arange(data_size)
                np.random.shuffle(indices)
                losses = []
                for start in range(0, data_size, BATCH_SIZE):
                    end = start + BATCH_SIZE
                    b_ind = indices[start:end]
                    _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(self.d_obs[b_ind], self.d_actions[b_ind])

                    logratio = new_logprobs - self.d_logprobs[b_ind]
                    ratio = torch.exp(logratio)

                    b_advantages = advantages_normal[b_ind]

                    objective_pg = torch.min(ratio * b_advantages, torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * b_advantages)
                    loss_pg = -objective_pg.mean()

                    loss_vf = 0.5 * ((new_values.flatten() - self.d_returns[b_ind])**2).mean()
                    
                    # Light action regularization
                    action_magnitude_penalty = torch.mean(torch.abs(self.d_actions[b_ind])) * JUMPING_PENALTY
                    
                    loss = loss_pg + loss_vf - ENT_COEF * entropy.mean() + action_magnitude_penalty
                    losses.append(loss.item())

                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                    self.optim.step()
            
            print(f"Run: {run}: Loss: {np.mean(losses):.4f}: Return: {returns[-1].mean().item():.4f}")

            if run % TEST_FREQ == 0:
                self.TestAgent(num_eval_episodes=5, run=run)
        self.env.close()



                
    def TestAgent(self, num_eval_episodes=5, run=0):
        print("----------------------------------")
        print(f"Agent Performance Test")
        print("----------------------------------")   
        frames = []
        for episode in range(num_eval_episodes):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            total_reward = 0
            done = False
            for i in range(TIMESTEPS):
                # obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    obs_norm = self.agent.normalize_obs(obs)
                    mean = self.agent.actor_net(obs_norm)
                    # mean = torch.tanh(mean)  # ensure actions are in [-1, 1]
                obs, reward, terminated, truncated, _ = self.env.step(mean)
                obs = obs["policy"]
                total_reward += reward
                frame = self.env.render()
                frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, f"Episode: {episode}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255,255,255), 2, cv2.LINE_AA)
                frames.append(frame)
                # if terminated or truncated:
                    # obs, info = self.env.reset()
                    # obs = obs["policy"]
                    # break
            print(f"Total reward: {total_reward.mean().item():.4f}")
        imageio.mimsave(f"outputs/{self.env_name}_{run}.mp4", frames, fps=30)
        torch.save(self.agent.state_dict(), f"outputs/{self.env_name}_{run}.pth")
        print("Video and Model Saved!!")



if __name__ == "__main__":
    # myenv = MyIsaacEnv("Isaac-Humanoid-v0")
    myenv = MyIsaacEnv("Isaac-Reach-Franka-v0")
    # myenv.TakeRandomActions()
    myenv.TrainAgent()
    simulation_app.close()