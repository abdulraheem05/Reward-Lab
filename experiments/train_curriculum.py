
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "models" / "curriculum_experiment"
LOG_DIR = ROOT / "logs" / "curriculum_experiment"

BEST_DIR = OUTPUT_DIR / "best"
EVAL_LOG_DIR = LOG_DIR / "evaluation"

for directory in (BEST_DIR, EVAL_LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
STEPS_PER_STAGE = 150_000

# Progressively increase gravity magnitude.
CURRICULUM = [
    {"name": "Easy", "gravity": -8.0},
    {"name": "Intermediate", "gravity": -9.0},
    {"name": "Default", "gravity": -10.0},
]


def make_env(gravity=-10.0):
    env = gym.make(
        "LunarLander-v3",
        gravity=gravity,
        enable_wind=False,
    )
    return Monitor(env)


# IMPORTANT: This is a fresh environment and a fresh policy.
# No baseline checkpoint is loaded.
train_env = make_env(gravity=CURRICULUM[0]["gravity"])

# Always evaluate on the standard environment.
eval_env = make_env(gravity=-10.0)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={"net_arch": [64, 64]},
    seed=SEED,
    verbose=1,
    tensorboard_log=str(LOG_DIR / "tensorboard"),
    device="auto",
)

# One callback throughout training, so the best score on the
# standard evaluation environment is tracked across all stages.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_DIR),
    log_path=str(EVAL_LOG_DIR),
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

print("\nStarting PPO curriculum learning FROM SCRATCH")
print("No pretrained model will be used.")
print(f"Total timesteps: {STEPS_PER_STAGE * len(CURRICULUM):,}")

for stage_index, stage in enumerate(CURRICULUM):
    print("\n" + "=" * 55)
    print(
        f"STAGE {stage_index + 1}/{len(CURRICULUM)}: "
        f"{stage['name']}"
    )
    print(f"Gravity: {stage['gravity']}")
    print(f"Timesteps: {STEPS_PER_STAGE:,}")
    print("=" * 55)

    if stage_index > 0:
        old_env = train_env
        train_env = make_env(gravity=stage["gravity"])

        # Keep the learned policy and optimizer, but change the
        # training environment. This is NOT loading the baseline.
        model.set_env(train_env)
        old_env.close()

    model.learn(
        total_timesteps=STEPS_PER_STAGE,
        callback=eval_callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    print(f"Completed stage: {stage['name']}")

final_path = OUTPUT_DIR / "final_model.zip"
model.save(str(final_path))

train_env.close()
eval_env.close()

print("\nCurriculum experiment complete.")
print(f"Final model: {final_path}")
print(f"Best model: {BEST_DIR / 'best_model.zip'}")
print(f"Evaluation logs: {EVAL_LOG_DIR}")
