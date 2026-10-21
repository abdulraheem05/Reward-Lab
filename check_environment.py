
import gymnasium as gym
import stable_baselines3
import torch

# Create the environment without opening a game window
env = gym.make("LunarLander-v3")

observation, info = env.reset(seed=42)

print("=== REWARDLAB ENVIRONMENT CHECK ===")
print(f"Observation: {observation}")
print(f"Observation shape: {observation.shape}")
print(f"Number of actions: {env.action_space.n}")
print(f"Action space: {env.action_space}")
print(f"Stable-Baselines3: {stable_baselines3.__version__}")
print(f"PyTorch: {torch.__version__}")

print("\n=== THE 8 OBSERVATIONS ===")

labels = [
    "Horizontal position",
    "Vertical position",
    "Horizontal velocity",
    "Vertical velocity",
    "Angle",
    "Angular velocity",
    "Left leg contact",
    "Right leg contact",
]

for label, value in zip(labels, observation):
    print(f"{label}: {value:.4f}")

env.close()
