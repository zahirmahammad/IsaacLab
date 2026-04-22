"""
PPO Training Script for Isaac Lab Environments
Trains a continuous control agent using Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import imageio
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict

from isaaclab.source.isaaclab.isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab.source.isaaclab_tasks.isaaclab_tasks.utils import load_cfg_from_registry

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    env_name: str = "Isaac-Lift-Cube-Franka-v0"
    device: str = "cuda"
    total_timesteps: int = int(2e7)
    
    # Environment
    num_envs: int = 256
    timesteps: int = 256
        
    # PPO
    learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 1024
    num_epochs: int = 5
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Regularization
    action_magnitude_penalty: float = 0.5
    
    # Misc
    test_frequency: float = 0.25  # Test 4 times during training
    output_dir: str = "outputs"


# ============================================================================
# NEURAL NETWORK AGENT
# ============================================================================

class PPOAgent(nn.Module):
    """Proximal Policy Optimization Agent with shared feature extraction"""
    
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Observation normalization buffers
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))
        
        # Actor network - outputs action mean
        self.actor_net = nn.Sequential(
            self._init_layer(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, action_dim), std=0.01),
        )
        
        # Actor log standard deviation (trainable)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network - outputs value estimate
        self.critic_net = nn.Sequential(
            self._init_layer(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            self._init_layer(nn.Linear(256, 128)),
            nn.Tanh(),
            self._init_layer(nn.Linear(128, 1), std=1.0),
        )
    
    @staticmethod
    def _init_layer(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        """Initialize linear layer with orthogonal weights"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def update_observation_stats(self, obs: torch.Tensor) -> None:
        """Update running mean and variance of observations using Welford's algorithm"""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count
    
    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics"""
        return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate from observations"""
        obs_norm = self.normalize_observation(obs)
        return self.critic_net(obs_norm)
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value from observation"""
        obs_norm = self.normalize_observation(obs)
        
        action_mean = self.actor_net(obs_norm)
        action_logstd = torch.clamp(self.actor_logstd, -5, 2)
        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy, self.get_value(obs)


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class PPOTrainer:
    """Proximal Policy Optimization Trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        Path(config.output_dir).mkdir(exist_ok=True)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize agent
        obs_dim = self.env.observation_space["policy"].shape[1]
        action_dim = self.env.action_space.shape[1]
        
        self.agent = PPOAgent(obs_dim, action_dim).to(config.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        
        print(f"✓ Environment: {config.env_name}")
        print(f"✓ Observation space: {obs_dim}")
        print(f"✓ Action space: {action_dim}")
        print(f"✓ Total parameters: {sum(p.numel() for p in self.agent.parameters()):,}")
    
    def _create_environment(self) -> gym.Env:
        """Create Isaac Lab environment with proper configuration"""
        cfg = load_cfg_from_registry(self.config.env_name, "env_cfg_entry_point")
        cfg.scene.num_envs = self.config.num_envs
        
        env = gym.make(self.config.env_name, cfg=cfg, render_mode="rgb_array")
        env = gym.wrappers.OrderEnforcing(env)
        return env
    
    def train(self) -> None:
        """Main training loop"""
        num_runs = int(self.config.total_timesteps // 
                      (self.config.timesteps * self.config.num_envs))
        test_freq = max(1, int(num_runs * self.config.test_frequency))
        
        done = torch.zeros(self.config.num_envs).to(self.config.device)
        
        # Preallocate rollout storage
        obs_buffer = torch.zeros((self.config.timesteps, self.config.num_envs, 
                                  self.agent.obs_dim)).to(self.config.device)
        action_buffer = torch.zeros((self.config.timesteps, self.config.num_envs, 
                                    self.agent.action_dim)).to(self.config.device)
        logprob_buffer = torch.zeros((self.config.timesteps, self.config.num_envs)).to(self.config.device)
        reward_buffer = torch.zeros((self.config.timesteps, self.config.num_envs)).to(self.config.device)
        done_buffer = torch.zeros((self.config.timesteps, self.config.num_envs)).to(self.config.device)
        value_buffer = torch.zeros((self.config.timesteps, self.config.num_envs)).to(self.config.device)
        
        print(f"\n{'='*60}")
        print(f"Training for {num_runs} runs ({self.config.total_timesteps/1e6:.0f}M timesteps)")
        print(f"{'='*60}\n")
        
        for run in range(num_runs):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            
            # Collect rollout
            for step in range(self.config.timesteps):
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs)
                
                obs_buffer[step] = obs
                action_buffer[step] = action
                done_buffer[step] = done
                value_buffer[step] = value.flatten()
                logprob_buffer[step] = logprob
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = obs["policy"]
                done = (terminated | truncated).float().to(self.config.device)
                reward_buffer[step] = reward
            
            # Update normalization statistics
            self.agent.update_observation_stats(obs_buffer.reshape(-1, self.agent.obs_dim))
            
            # Compute advantages and returns
            advantages, returns = self._compute_advantages_and_returns(
                obs, done, reward_buffer, done_buffer, value_buffer)
            
            # Perform PPO update
            loss = self._ppo_update(obs_buffer, action_buffer, logprob_buffer, 
                                    advantages, returns)
            
            # Logging
            print(f"Run {run+1:4d}/{num_runs} | Loss: {loss:.4f} | Return: {returns.mean():.4f}")
            
            # Periodic evaluation
            if (run + 1) % test_freq == 0:
                self.evaluate(run + 1)
        
        self.env.close()
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
    
    def _compute_advantages_and_returns(self, next_obs: torch.Tensor, done: torch.Tensor,
                                       reward_buffer, done_buffer, value_buffer
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(reward_buffer).to(self.config.device)
        lastgaelam = torch.zeros(self.config.num_envs).to(self.config.device)
        
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).flatten()
        
        for step in reversed(range(self.config.timesteps)):
            if step == self.config.timesteps - 1:
                next_done = 1 - done
                next_val = next_value
            else:
                next_done = 1 - done_buffer[step + 1]
                next_val = value_buffer[step + 1]
            
            delta = reward_buffer[step] + self.config.gamma * next_val * next_done - value_buffer[step]
            lastgaelam = delta + self.config.gamma * self.config.gae_lambda * next_done * lastgaelam
            advantages[step] = lastgaelam
        
        returns = advantages + value_buffer
        return advantages, returns
    
    def _ppo_update(self, obs_buffer, action_buffer, logprob_buffer, advantages, returns
                   ) -> float:
        """Perform PPO update with mini-batch gradient descent"""
        # Flatten buffers
        flat_obs = obs_buffer.reshape(-1, self.agent.obs_dim).to(self.config.device)
        flat_actions = action_buffer.reshape(-1, self.agent.action_dim).to(self.config.device)
        flat_logprobs = logprob_buffer.reshape(-1).to(self.config.device)
        flat_advantages = advantages.reshape(-1).to(self.config.device)
        flat_returns = returns.reshape(-1).to(self.config.device)
        
        # Normalize advantages
        advantages_normalized = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        data_size = self.config.num_envs * self.config.timesteps
        losses = []
        
        for epoch in range(self.config.num_epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, data_size, self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(
                    flat_obs[batch_indices], flat_actions[batch_indices])
                
                logratio = new_logprobs - flat_logprobs[batch_indices]
                ratio = torch.exp(logratio)
                
                batch_advantages = advantages_normalized[batch_indices]
                
                # Policy loss with clipping
                objective_pg = torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages
                )
                loss_pg = -objective_pg.mean()
                
                # Value function loss
                loss_vf = 0.5 * ((new_values.flatten() - flat_returns[batch_indices])**2).mean()
                
                # Action magnitude penalty (discourages explosive movements)
                action_penalty = torch.mean(torch.abs(flat_actions[batch_indices])) * self.config.action_magnitude_penalty
                
                # Total loss
                total_loss = loss_pg + loss_vf - self.config.entropy_coef * entropy.mean() + action_penalty
                losses.append(total_loss.item())
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        return np.mean(losses)
    
    def evaluate(self, run: int) -> None:
        """Evaluate trained agent"""
        print(f"\n{'-'*60}")
        print(f"Evaluation at Run {run}")
        print(f"{'-'*60}")
        
        frames = []
        for episode in range(5):
            obs, _ = self.env.reset()
            obs = obs["policy"]
            total_reward = 0
            
            for _ in range(self.config.timesteps):
                with torch.no_grad():
                    obs_norm = self.agent.normalize_observation(obs)
                    action = self.agent.actor_net(obs_norm)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = obs["policy"]
                total_reward += reward
                
                frame = self.env.render()
                frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, f"Episode: {episode}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                frames.append(frame)
            
            print(f"Episode {episode+1}: Reward = {total_reward.mean().item():.4f}")
        
        # Save video and checkpoint
        video_path = f"{self.config.output_dir}/{self.config.env_name}_{run}.mp4"
        model_path = f"{self.config.output_dir}/{self.config.env_name}_{run}.pth"
        
        imageio.mimsave(video_path, frames, fps=30)
        torch.save(self.agent.state_dict(), model_path)
        
        print(f"✓ Video saved: {video_path}")
        print(f"✓ Model saved: {model_path}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    # Configure and train
    config = TrainingConfig()
    trainer = PPOTrainer(config)
    trainer.train()
    
    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
