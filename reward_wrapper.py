
import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Reward experiments:
    baseline      - use the original reward unchanged
    time_penalty  - charge a small cost per frame
    hover_bonus   - deliberately reward hovering-like states
    progress      - add a distance-progress shaping term
    """

    def __init__(
        self,
        env,
        mode="baseline",
        time_cost=0.05,
        hover_bonus=1.0,
        progress_scale=1.0,
    ):
        super().__init__(env)

        valid_modes = {
            "baseline",
            "time_penalty",
            "hover_bonus",
            "progress",
        }

        if mode not in valid_modes:
            raise ValueError(f"Unknown reward mode: {mode}")

        self.mode = mode
        self.time_cost = time_cost
        self.hover_bonus = hover_bonus
        self.progress_scale = progress_scale
        self.previous_distance = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        x, y = observation[0], observation[1]
        self.previous_distance = float(np.sqrt(x**2 + y**2))

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Individual reward components for later analysis
        custom_reward = 0.0
        x, y, vx, vy, angle, angular_velocity = observation[:6]

        distance = float(np.sqrt(x**2 + y**2))

        if self.mode == "time_penalty":
            custom_reward -= self.time_cost

        elif self.mode == "hover_bonus":
            # Intentionally flawed: rewards slow, upright states
            # even when the rocket is not making landing progress.
            hovering = (
                abs(angle) < 0.1
                and abs(vx) < 0.1
                and abs(vy) < 0.1
                and not terminated
            )

            if hovering:
                custom_reward += self.hover_bonus

        elif self.mode == "progress":
            # Reward a reduction in distance to the landing area.
            # This is an experimental heuristic, not a guarantee
            # of better landing performance.
            if self.previous_distance is not None:
                custom_reward += self.progress_scale * (
                    self.previous_distance - distance
                )

        self.previous_distance = distance

        # Keep the original reward and add our experimental term.
        modified_reward = float(reward + custom_reward)

        info = dict(info)
        info["original_reward"] = float(reward)
        info["custom_reward"] = float(custom_reward)

        return (
            observation,
            modified_reward,
            terminated,
            truncated,
            info,
        )
