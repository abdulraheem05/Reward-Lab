
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# Project directories
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Training configuration
TOTAL_TIMESTEPS = 300_000
SEED = 42

# Training environment
train_env = Monitor(gym.make("LunarLander-v3"))

# Separate environment for periodic evaluation
eval_env = Monitor(gym.make("LunarLander-v3"))

# Save the best model based on evaluation performance
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODEL_DIR / "best"),
    log_path=str(LOG_DIR / "evaluation"),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1,
)

# PPO with a small neural network policy
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
    policy_kwargs=dict(net_arch=[64, 64]),
    seed=SEED,
    verbose=1,
    tensorboard_log=str(LOG_DIR / "tensorboard"),
    device="auto",
)

print("\nStarting PPO training...")
print(f"Training steps: {TOTAL_TIMESTEPS:,}")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True,
)

# Save the final model
model.save(str(MODEL_DIR / "ppo_lunar_lander"))

print("\nTraining complete!")
print("Final model: models/ppo_lunar_lander.zip")
print("Best checkpoint: models/best/best_model.zip")

train_env.close()
eval_env.close()
