
import gymnasium as gym
from stable_baselines3 import PPO

from reward_wrapper import RewardShapingWrapper

model = PPO.load(
    "models/experiments/hover_bonus/model.zip"
)

# Original environment: no reward wrapper during evaluation
env = gym.make(
    "LunarLander-v3",
    render_mode="human",
)

for episode in range(3):
    obs, info = env.reset(seed=200 + episode)
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(
        f"Episode {episode + 1}: "
        f"reward={total_reward:.2f}, "
        f"steps={steps}"
    )

env.close()
