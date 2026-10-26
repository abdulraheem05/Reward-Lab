
import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mode="baseline",
        time_cost=0.05,
        hover_bonus=1.0,
        progress_scale=1.0,
        potential_scale=0.1,
        gamma=0.99,
    ):
        super().__init__(env)

        valid_modes = {
            "baseline",
            "time_penalty",
            "hover_bonus",
            "progress",
            "potential",
        }

        if mode not in valid_modes:
            raise ValueError(f"Unknown reward mode: {mode}")

        self.mode = mode
        self.time_cost = time_cost
        self.hover_bonus = hover_bonus
        self.progress_scale = progress_scale
        self.potential_scale = potential_scale
        self.gamma = gamma

        self.previous_distance = None
        self.previous_potential = None

    @staticmethod
    def potential(observation):
        """
        Higher potential means a more desirable landing state.
        Observation values are supplied by LunarLander-v3.
        """
        x, y, vx, vy, angle, angular_velocity = observation[:6]

        distance = np.sqrt(x**2 + y**2)
        speed = np.sqrt(vx**2 + vy**2)

        # Negative weighted cost: closer, slower and more
        # upright states have higher potential.
        return -(
            0.5 * distance
            + 0.3 * speed
            + 0.2 * abs(angle)
            + 0.05 * abs(angular_velocity)
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        x, y = observation[:2]
        self.previous_distance = float(np.sqrt(x**2 + y**2))
        self.previous_potential = self.potential(observation)

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        custom_reward = 0.0

        x, y, vx, vy, angle, angular_velocity = observation[:6]
        distance = float(np.sqrt(x**2 + y**2))

        if self.mode == "time_penalty":
            custom_reward = -self.time_cost

        elif self.mode == "hover_bonus":
            hovering = (
                abs(angle) < 0.1
                and abs(vx) < 0.1
                and abs(vy) < 0.1
                and not terminated
            )

            if hovering:
                custom_reward = self.hover_bonus

        elif self.mode == "progress":
            if self.previous_distance is not None:
                custom_reward = self.progress_scale * (
                    self.previous_distance - distance
                )

        elif self.mode == "potential":
            next_potential = self.potential(observation)

            # A true terminal state has zero potential.
            # Truncation is kept separate from termination.
            if terminated:
                next_potential = 0.0

            custom_reward = self.potential_scale * (
                self.gamma * next_potential
                - self.previous_potential
            )

            self.previous_potential = self.potential(observation)

        self.previous_distance = distance

        info = dict(info)
        info["original_reward"] = float(reward)
        info["custom_reward"] = float(custom_reward)

        return (
            observation,
            float(reward + custom_reward),
            terminated,
            truncated,
            info,
        )
