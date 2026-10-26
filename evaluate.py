
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


parser = argparse.ArgumentParser()
parser.add_argument(
    "--render",
    action="store_true",
    help="Open the LunarLander game window",
)
parser.add_argument("--episodes", type=int, default=100)
args = parser.parse_args()

# Prefer the best checkpoint from training
best_path = Path("models/best/best_model.zip")
final_path = Path("models/fine_tuned/final_model.zip")

if best_path.exists():
    model_path = best_path
elif final_path.exists():
    model_path = final_path
else:
    raise FileNotFoundError(
        "No trained model found. Run python train.py first."
    )

model = PPO.load(str(model_path))
env = gym.make(
    "LunarLander-v3",
    render_mode="human" if args.render else None,
)

episode_rewards = []

print(f"Evaluating: {model_path}")
print(f"Episodes: {args.episodes}\n")

for episode in range(1, args.episodes + 1):
    observation, info = env.reset(seed=1000 + episode)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Deterministic prediction: choose the policy's preferred action
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode:02d}: reward = {total_reward:.2f}")

env.close()

rewards = np.array(episode_rewards)
threshold_count = int(np.sum(rewards >= 200))

print("\n=== EVALUATION SUMMARY ===")
print(f"Mean reward: {rewards.mean():.2f}")
print(f"Standard deviation: {rewards.std():.2f}")
print(
    f"Episodes scoring >= 200: "
    f"{threshold_count}/{len(rewards)} "
    f"({100 * threshold_count / len(rewards):.1f}%)"
)
