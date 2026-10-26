
from pathlib import Path
import re

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos" / "model_recordings"

MODELS = {
    "Baseline PPO": "models/best/best_model.zip",

    "Time Penalty": "models/experiments/time_penalty/model.zip",
    "Hover Bonus": "models/experiments/hover_bonus/model.zip",
    "Progress Shaping": "models/experiments/progress/model.zip",
    "Potential Shaping": "models/experiments/potential/model.zip",

    "Fine-tuned PPO - Best": "models/fine_tuned/best/best_model.zip",
    "Fine-tuned PPO - Final": "models/fine_tuned/final_model.zip",

    "Curriculum PPO - Best": (
        "models/curriculum_experiment/best/best_model.zip"
    ),
    "Curriculum PPO - Final": (
        "models/curriculum_experiment/final_model.zip"
    ),
}

SEED = 2026
MAX_STEPS = 2000


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def record_model(name, model_path):
    print(f"\nRecording: {name}")

    model = PPO.load(str(model_path), device="auto")

    output_folder = VIDEO_DIR / safe_name(name)
    output_folder.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
    )

    env = RecordVideo(
        env,
        video_folder=str(output_folder),
        episode_trigger=lambda episode_id: True,
        name_prefix=safe_name(name),
        disable_logger=True,
    )

    try:
        obs, info = env.reset(seed=SEED)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            if steps >= MAX_STEPS:
                break

        print(f"Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Video folder: {output_folder}")

    finally:
        env.close()


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    recorded = []
    skipped = []

    for name, relative_path in MODELS.items():
        model_path = ROOT / relative_path

        if not model_path.exists():
            print(f"SKIPPING {name}: checkpoint not found")
            skipped.append(name)
            continue

        record_model(name, model_path)
        recorded.append(name)

    print("\n" + "=" * 50)
    print(f"Videos recorded: {len(recorded)}")
    print(f"Models skipped: {len(skipped)}")
    print(f"Output directory: {VIDEO_DIR}")

    if skipped:
        print("Skipped models:", ", ".join(skipped))


if __name__ == "__main__":
    main()
