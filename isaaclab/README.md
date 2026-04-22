# MyScripts - PPO Training for Isaac Lab Environments

This folder contains scripts for training reinforcement learning agents using **Proximal Policy Optimization (PPO)** in Isaac Lab simulation environments. The scripts demonstrate how to train AI agents to perform complex robotic manipulation tasks.

## 📚 What Is This?

PPO is a modern reinforcement learning algorithm that trains neural networks to control robots in physics simulations. The scripts in this folder:

- **Train agents** to learn from experience and improve their performance over time
- **Simulate multiple environments in parallel** (256+ environments at once for faster training)
- **Record videos** of the trained agents performing tasks
- **Save trained models** for evaluation and deployment

## 📁 Files

- **`MyPPO_Isaac.py`** - Main training script with configurable PPO trainer for any Isaac Lab environment
- **`isaac_env_ppo.py`** - PPO implementation with detailed training loop
- **`testscript.py`** - Basic example script for testing individual environments
- **`run_envs.sh`** - Bash script to run training on multiple environments

## 🤖 Trained Environments

Here are 6 different robotic manipulation tasks trained with PPO. Each video shows the agent at a specific training checkpoint:

### Training Progress Visualization (Video at checkpoint 304)

<table>
<tr>
<td align="center">
<b>Lift Cube - Franka</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Lift-Cube-Franka-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>Franka robot lifting a cube off the ground</i>
</td>
<td align="center">
<b>Lift Cube - OpenArm</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Lift-Cube-OpenArm-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>OpenArm robot lifting a cube off the ground</i>
</td>
<td align="center">
<b>Open Drawer - Franka</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Open-Drawer-Franka-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>Franka robot opening a drawer</i>
</td>
</tr>
<tr>
<td align="center">
<b>Open Drawer - OpenArm</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Open-Drawer-OpenArm-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>OpenArm robot opening a drawer</i>
</td>
<td align="center">
<b>Reach Franka</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Reach-Franka-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>Franka robot reaching for a target location</i>
</td>
<td align="center">
<b>Reach OpenArm</b>
<br/>
<video width="200" height="200" controls>
<source src="outputs/Isaac-Reach-OpenArm-v0/videos/304.mp4" type="video/mp4">
</video>
<br/>
<i>OpenArm robot reaching for a target location</i>
</td>
</tr>
</table>

## 🚀 Quick Start

### Run Training

```bash
cd /home/zahir/IsaacLab
python myscripts/MyPPO_Isaac.py --env Isaac-Reach-Franka-v0 --total-timesteps 20000000
```

### Test a Trained Model

```bash
python myscripts/testscript.py
```

## 📊 Key Training Parameters

- **Learning Rate**: 5e-5
- **Parallel Environments**: 256 (for fast data collection)
- **Total Timesteps**: 20 million (20M steps of experience)
- **PPO Epochs**: 10 passes through collected data each iteration
- **Batch Size**: 1024 samples per gradient update
- **Discount Factor (γ)**: 0.99
- **GAE Lambda**: 0.95

## 💾 Output Structure

Each environment creates a folder in `outputs/` with:

```
outputs/Isaac-Reach-Franka-v0/
├── videos/
│   ├── 76.mp4      # Video at training checkpoint 76
│   ├── 152.mp4     # Video at training checkpoint 152
│   ├── 228.mp4     # Video at training checkpoint 228
│   └── 304.mp4     # Video at training checkpoint 304
└── models/
    ├── 76.pth      # Trained model weights at checkpoint 76
    ├── 152.pth
    ├── 228.pth
    └── 304.pth
```

## 🔧 How PPO Works (Simple Explanation)

1. **Collect Experience**: Run 256 robots in parallel for 256 steps each
2. **Calculate Rewards**: Measure how well each action achieved the goal
3. **Update Policy**: Improve the neural network using the collected experience
4. **Repeat**: Do this ~5000 times until the agent learns the task

## 📈 Training Tips

- **GPU Required**: NVIDIA GPU (CUDA) recommended for fast training
- **Patience**: 20M steps takes hours to days depending on your GPU
- **Hyperparameter Tuning**: Adjust learning rate, entropy coefficient, and action penalties based on task
- **Monitor Progress**: Watch the test videos to see if the agent is improving

## 🤝 Dependencies

- **Isaac Lab** - Physics simulation and environments
- **PyTorch** - Neural network framework
- **Gymnasium** - RL environment interface
- **OpenCV** - Video processing

---

**Status**: ✅ Successfully trained on 6 robotic manipulation tasks
