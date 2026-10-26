
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parent.parent

# Start from your strongest existing baseline checkpoint.
BASELINE_PATH = ROOT / "models" / "best" / "best_model.zip"

OUTPUT_DIR = ROOT / "models" / "fine_tuned"
LOG_DIR = ROOT / "logs" / "fine_tuned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ADDITIONAL_STEPS = 100_000
LEARNING_RATE = 1e-5


def make_env():
    # Preserve the original LunarLander reward.
    return Monitor(gym.make("LunarLander-v3"))


if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Baseline checkpoint not found: {BASELINE_PATH}"
    )

train_env = make_env()
eval_env = make_env()

# Continue learning from the existing trained policy.
model = PPO.load(
    str(BASELINE_PATH),
    env=train_env,
    learning_rate=LEARNING_RATE,
    device="auto",
)

# Use the lower learning rate for conservative updates.
for group in model.policy.optimizer.param_groups:
    group["lr"] = LEARNING_RATE

# Save the best checkpoint based on the ORIGINAL environment.
callback = EvalCallback(
    eval_env,
    best_model_save_path=str(OUTPUT_DIR / "best"),
    log_path=str(LOG_DIR / "evaluation"),
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

print("Starting conservative PPO fine-tuning")
print(f"Baseline: {BASELINE_PATH}")
print(f"Additional steps: {ADDITIONAL_STEPS:,}")
print(f"Learning rate: {LEARNING_RATE}")

model.learn(
    total_timesteps=ADDITIONAL_STEPS,
    callback=callback,
    reset_num_timesteps=False,
    progress_bar=True,
)

model.save(str(OUTPUT_DIR / "final_model"))

train_env.close()
eval_env.close()

print("\nExperiment complete.")
print(f"Best model: {OUTPUT_DIR / 'best' / 'best_model.zip'}")
print(f"Final model: {OUTPUT_DIR / 'final_model.zip'}")
