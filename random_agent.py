
import gymnasium as gym

# Open the LunarLander game window
env = gym.make("LunarLander-v3", render_mode="human")

number_of_episodes = 5

for episode in range(1, number_of_episodes + 1):
    observation, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        # Choose a random action
        action = env.action_space.sample()

        # Apply the action to the simulation
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(
        f"Episode: {episode} | "
        f"Steps: {steps} | "
        f"Total reward: {total_reward:.2f}"
    )

env.close()
