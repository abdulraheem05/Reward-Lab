
import json
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent.parent

EPISODES = 100
BASE_SEED = 1000
SUCCESS_THRESHOLD = 200

# Evaluate every model on the original, unmodified environment.
MODELS = {
    "Baseline PPO": "models/best/best_model.zip",

    "Time Penalty": "models/experiments/time_penalty/model.zip",
    "Hover Bonus": "models/experiments/hover_bonus/model.zip",
    "Progress": "models/experiments/progress/model.zip",
    "Potential Shaping": "models/experiments/potential/model.zip",

    "Fine-tuned PPO - Best": (
        "models/fine_tuned/best/best_model.zip"
    ),
    "Fine-tuned PPO - Final": (
        "models/fine_tuned/final_model.zip"
    ),
}


def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def evaluate_model(name, relative_path):
    model_path = ROOT / relative_path

    if not model_path.exists():
        print(f"\nSKIPPING {name}: checkpoint not found")
        print(f"Expected path: {model_path}")
        return None, None

    print("\n" + "=" * 60)
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {relative_path}")
    print(f"Episodes: {EPISODES}")
    print("=" * 60)

    model = PPO.load(str(model_path), device="auto")
    env = gym.make("LunarLander-v3")

    episode_rows = []

    try:
        for episode_number in range(1, EPISODES + 1):
            seed = BASE_SEED + episode_number

            observation, info = env.reset(seed=seed)

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

                total_reward += float(reward)
                steps += 1

            row = {
                "model": name,
                "checkpoint": relative_path,
                "episode": episode_number,
                "seed": seed,
                "reward": total_reward,
                "episode_length": steps,
                "success_ge_200": (
                    total_reward >= SUCCESS_THRESHOLD
                ),
                "reward_below_0": total_reward < 0,
                "terminated": terminated,
                "truncated": truncated,
            }

            episode_rows.append(row)

            print(
                f"Episode {episode_number:03d}/{EPISODES}: "
                f"reward={total_reward:8.2f}, steps={steps}"
            )

    finally:
        env.close()

    episodes_df = pd.DataFrame(episode_rows)
    rewards = episodes_df["reward"].to_numpy()
    lengths = episodes_df["episode_length"].to_numpy()

    summary = {
        "model": name,
        "checkpoint": relative_path,
        "evaluation_episodes": len(rewards),
        "base_seed": BASE_SEED,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_threshold": SUCCESS_THRESHOLD,
        "episodes_ge_200": int(np.sum(rewards >= SUCCESS_THRESHOLD)),
        "success_rate_pct": float(
            100 * np.mean(rewards >= SUCCESS_THRESHOLD)
        ),
        "episodes_below_0": int(np.sum(rewards < 0)),
        "mean_episode_length": float(np.mean(lengths)),
    }

    print("\nSUMMARY")
    print(f"Mean reward: {summary['mean_reward']:.2f}")
    print(f"Reward std: {summary['std_reward']:.2f}")
    print(
        f"Success: {summary['episodes_ge_200']}/"
        f"{summary['evaluation_episodes']} "
        f"({summary['success_rate_pct']:.1f}%)"
    )

    return summary, episode_rows


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        ROOT / "results" / "evaluations" / f"run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_episodes = []
    skipped_models = []

    for name, relative_path in MODELS.items():
        summary, episodes = evaluate_model(name, relative_path)

        if summary is None:
            skipped_models.append({
                "model": name,
                "checkpoint": relative_path,
            })
            continue

        all_summaries.append(summary)
        all_episodes.extend(episodes)

    if not all_summaries:
        print("No models were evaluated. Check your checkpoint paths.")
        return

    summary_df = pd.DataFrame(all_summaries)
    episode_df = pd.DataFrame(all_episodes)

    # Save every individual episode.
    episode_df.to_csv(
        output_dir / "episode_results.csv",
        index=False,
    )

    # Save aggregate results for easy comparison.
    summary_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    metadata = {
        "evaluation_timestamp": timestamp,
        "environment": "LunarLander-v3",
        "environment_reward": "Original Gymnasium reward",
        "episodes_per_model": EPISODES,
        "base_seed": BASE_SEED,
        "seeds": list(range(BASE_SEED + 1, BASE_SEED + EPISODES + 1)),
        "deterministic_actions": True,
        "success_threshold": SUCCESS_THRESHOLD,
        "evaluated_models": [
            {
                "model": row["model"],
                "checkpoint": row["checkpoint"],
            }
            for row in all_summaries
        ],
        "skipped_models": skipped_models,
        "software_versions": {
            "gymnasium": package_version("gymnasium"),
            "stable-baselines3": package_version(
                "stable-baselines3"
            ),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
    }

    with open(
        output_dir / "evaluation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    columns = [
        "model",
        "mean_reward",
        "std_reward",
        "success_rate_pct",
        "mean_episode_length",
        "episodes_below_0",
    ]

    print(
        summary_df[columns].round(2).to_string(index=False)
    )

    print("\nResults saved to:")
    print(output_dir)
    print("\nFiles created:")
    print("  episode_results.csv")
    print("  model_comparison.csv")
    print("  evaluation_metadata.json")

    if skipped_models:
        print("\nSome checkpoints were skipped.")
        print("Check the paths printed above before finalizing results.")


if __name__ == "__main__":
    main()
