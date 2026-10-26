
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


MODELS = {
    "Baseline PPO": "models/best/best_model.zip",
    "Time Penalty": "models/experiments/time_penalty/model.zip",
    "Hover Bonus": "models/experiments/hover_bonus/model.zip",
    "Progress": "models/experiments/progress/model.zip",
    "Potential Shaping": "models/experiments/potential/model.zip",
}

EPISODES = 100
BASE_SEED = 1000


def evaluate_model(name, model_path):
    path = Path(model_path)

    if not path.exists():
        print(f"\nSkipping {name}: model not found at {path}")
        return None

    # IMPORTANT: no custom reward wrapper here.
    env = gym.make("LunarLander-v3")
    model = PPO.load(str(path))

    rewards = []
    lengths = []

    print(f"\nEvaluating {name}...")

    for episode in range(EPISODES):
        # Use matching seeds across models for a fair comparison.
        observation, info = env.reset(seed=BASE_SEED + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    env.close()

    rewards = np.asarray(rewards)
    lengths = np.asarray(lengths)

    results = {
        "model": name,
        "mean_reward": rewards.mean(),
        "std_reward": rewards.std(),
        "success_rate_pct": 100 * np.mean(rewards >= 200),
        "episodes_ge_200": int(np.sum(rewards >= 200)),
        "mean_episode_length": lengths.mean(),
        "mean_steps": lengths.mean(),
        "episodes_below_0": int(np.sum(rewards < 0)),
    }

    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Reward std: {results['std_reward']:.2f}")
    print(
        f"Score >= 200: {results['episodes_ge_200']}"
        f"/{EPISODES} ({results['success_rate_pct']:.1f}%)"
    )
    print(f"Mean episode length: {results['mean_episode_length']:.1f}")
    print(f"Episodes below 0 reward: {results['episodes_below_0']}")

    return results


def main():
    all_results = []

    for name, path in MODELS.items():
        result = evaluate_model(name, path)

        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo models were evaluated. Check the model paths.")
        return

    df = pd.DataFrame(all_results)
    output_path = Path("experiments")
    output_path.mkdir(exist_ok=True)

    df.to_csv(output_path / "evaluation_results.csv", index=False)

    print("\n=== FINAL COMPARISON ===")
    print(
        df[
            [
                "model",
                "mean_reward",
                "success_rate_pct",
                "mean_episode_length",
                "episodes_below_0",
            ]
        ].to_string(index=False)
    )

    print("\nSaved: experiments/evaluation_results.csv")


if __name__ == "__main__":
    main()
