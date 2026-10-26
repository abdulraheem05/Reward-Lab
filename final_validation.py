
from pathlib import Path
from datetime import datetime
import json

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODELS = {
    "Baseline PPO": ROOT / "models" / "best" / "best_model.zip",
    "Fine-tuned PPO - Final": (
        ROOT / "models" / "fine_tuned" / "final_model.zip"
    ),
}

EPISODES_PER_SET = 100

# Three separate sets of environment seeds.
SEED_SETS = {
    "Set 1": list(range(2001, 2101)),
    "Set 2": list(range(3001, 3101)),
    "Set 3": list(range(4001, 4101)),
}

SUCCESS_THRESHOLD = 200
BOOTSTRAP_SAMPLES = 10_000


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_episode(model, env, seed, model_name, set_name):
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

    # LunarLander observation:
    # x, y, vx, vy, angle, angular_velocity,
    # left_leg_contact, right_leg_contact
    final_obs = np.asarray(obs, dtype=float)

    return {
        "set": set_name,
        "seed": seed,
        "model": model_name,
        "reward": total_reward,
        "episode_length": steps,
        "success_ge_200": total_reward >= SUCCESS_THRESHOLD,
        "terminated": terminated,
        "truncated": truncated,
        "final_x": final_obs[0],
        "final_y": final_obs[1],
        "final_vx": final_obs[2],
        "final_vy": final_obs[3],
        "final_angle": final_obs[4],
        "final_angular_velocity": final_obs[5],
        "left_leg_contact": bool(final_obs[6] > 0.5),
        "right_leg_contact": bool(final_obs[7] > 0.5),
    }


def summarize(df):
    rows = []

    for (set_name, model_name), group in df.groupby(
        ["set", "model"], sort=False
    ):
        rewards = group["reward"].to_numpy()

        rows.append({
            "set": set_name,
            "model": model_name,
            "episodes": len(group),
            "mean_reward": rewards.mean(),
            "std_reward": rewards.std(ddof=1),
            "median_reward": np.median(rewards),
            "min_reward": rewards.min(),
            "max_reward": rewards.max(),
            "successes_ge_200": int(
                group["success_ge_200"].sum()
            ),
            "success_rate_pct": (
                100 * group["success_ge_200"].mean()
            ),
            "mean_episode_length": group["episode_length"].mean(),
        })

    # Overall results across all three sets.
    for model_name, group in df.groupby("model", sort=False):
        rewards = group["reward"].to_numpy()

        rows.append({
            "set": "OVERALL",
            "model": model_name,
            "episodes": len(group),
            "mean_reward": rewards.mean(),
            "std_reward": rewards.std(ddof=1),
            "median_reward": np.median(rewards),
            "min_reward": rewards.min(),
            "max_reward": rewards.max(),
            "successes_ge_200": int(
                group["success_ge_200"].sum()
            ),
            "success_rate_pct": (
                100 * group["success_ge_200"].mean()
            ),
            "mean_episode_length": group["episode_length"].mean(),
        })

    return pd.DataFrame(rows)


def paired_comparison(df):
    # Each model faces the same seed in every set.
    wide = df.pivot(
        index=["set", "seed"],
        columns="model",
        values="reward",
    ).dropna()

    baseline = wide["Baseline PPO"].to_numpy()
    candidate = wide["Fine-tuned PPO - Final"].to_numpy()

    differences = candidate - baseline
    rng = np.random.default_rng(20261026)

    # Bootstrap confidence interval for the paired mean difference.
    indices = rng.integers(
        0, len(differences),
        size=(BOOTSTRAP_SAMPLES, len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)

    ci_low, ci_high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    baseline_success = baseline >= SUCCESS_THRESHOLD
    candidate_success = candidate >= SUCCESS_THRESHOLD

    return {
        "matched_episodes": len(differences),
        "mean_reward_difference_candidate_minus_baseline": (
            differences.mean()
        ),
        "median_reward_difference": np.median(differences),
        "bootstrap_95pct_ci_lower": ci_low,
        "bootstrap_95pct_ci_upper": ci_high,
        "candidate_wins_by_reward": int(
            np.sum(differences > 0)
        ),
        "baseline_wins_by_reward": int(
            np.sum(differences < 0)
        ),
        "ties_by_reward": int(np.sum(differences == 0)),
        "baseline_successes_ge_200": int(
            baseline_success.sum()
        ),
        "candidate_successes_ge_200": int(
            candidate_success.sum()
        ),
        "baseline_success_rate_pct": (
            100 * baseline_success.mean()
        ),
        "candidate_success_rate_pct": (
            100 * candidate_success.mean()
        ),
        "success_rate_difference_percentage_points": (
            100 * (
                candidate_success.mean()
                - baseline_success.mean()
            )
        ),
    }


def main():
    missing = [
        str(path) for path in MODELS.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model checkpoint(s):\n" + "\n".join(missing)
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / "final_validation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("FINAL VALIDATION")
    print("Environment: LunarLander-v3")
    print("Episodes per set: 100")
    print("Sets: 3")
    print("Episodes per model: 300")
    print("Models: 2")
    print("Total episodes: 600")
    print("Actions: deterministic")
    print(f"Output: {output_dir}")

    # Load each checkpoint once.
    loaded_models = {
        name: PPO.load(str(path), device="auto")
        for name, path in MODELS.items()
    }

    all_rows = []

    # Fresh environment for each model/set.
    # Both models receive the same seed for every matched pair.
    for set_name, seeds in SEED_SETS.items():
        print(f"\n{'=' * 55}\n{set_name}\n{'=' * 55}")

        for model_name, model in loaded_models.items():
            env = gym.make("LunarLander-v3")

            try:
                for i, seed in enumerate(seeds, start=1):
                    row = evaluate_episode(
                        model, env, seed, model_name, set_name
                    )
                    all_rows.append(row)

                    print(
                        f"{model_name} | {i:03d}/{len(seeds)} "
                        f"| seed={seed} "
                        f"| reward={row['reward']:.2f}"
                    )
            finally:
                env.close()

    episode_df = pd.DataFrame(all_rows)
    summary_df = summarize(episode_df)
    comparison = paired_comparison(episode_df)

    episode_df.to_csv(
        output_dir / "episode_results.csv", index=False
    )
    summary_df.to_csv(
        output_dir / "summary.csv", index=False
    )

    with open(
        output_dir / "paired_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(comparison, f, indent=2)

    metadata = {
        "environment": "LunarLander-v3",
        "episodes_per_set_per_model": EPISODES_PER_SET,
        "seed_sets": SEED_SETS,
        "deterministic_actions": True,
        "success_threshold": SUCCESS_THRESHOLD,
        "models": {
            name: str(path.relative_to(ROOT))
            for name, path in MODELS.items()
        },
        "note": (
            "Success is operationally defined as episode reward >= 200; "
            "this is not a direct physical safe-landing test."
        ),
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 65)
    print("SUMMARY BY SET AND OVERALL")
    print("=" * 65)
    print(summary_df.round(2).to_string(index=False))

    print("\n" + "=" * 65)
    print("PAIRED COMPARISON: FINE-TUNED MINUS BASELINE")
    print("=" * 65)

    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved results to:")
    print(output_dir)
    print("  episode_results.csv")
    print("  summary.csv")
    print("  paired_comparison.json")
    print("  metadata.json")


if __name__ == "__main__":
    main()
