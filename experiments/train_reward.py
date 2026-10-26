
import argparse
from pathlib import Path
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reward_wrapper import RewardShapingWrapper


parser = argparse.ArgumentParser()
parser.add_argument(
    "--reward",
    choices=["baseline", "time_penalty", "hover_bonus", "progress", "potential"],
    required=True,
)
parser.add_argument("--steps", type=int, default=100_000)
args = parser.parse_args()

reward_name = args.reward

model_dir = ROOT /"models"/"experiments"/ reward_name
log_dir = ROOT /"logs"/ "experiments" / reward_name

model_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)


def make_env():
    env = gym.make("LunarLander-v3")
    env = RewardShapingWrapper(env, mode=reward_name)
    env = Monitor(env)
    return env


env = make_env()

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={"net_arch": [64, 64]},
    seed=42,
    verbose=1,
    tensorboard_log=str(log_dir / "tensorboard"),
    device="auto",
)

print(f"Training reward experiment: {reward_name}")
print(f"Training steps: {args.steps:,}")

model.learn(
    total_timesteps=args.steps,
    tb_log_name=reward_name,
    progress_bar=True,
)

model.save(str(model_dir / "model"))

print(f"Saved model: {model_dir / 'model.zip'}")

env.close()
