import gymnasium as gym

from gymnasium.envs.registration import register
from render.render import render_browser

register(
    id='morl/FrozenLakePlus-v1',
    entry_point='environments.frozen_lake_plus.frozen_lake_plus:FrozenLakePlusEnv',
    max_episode_steps=300,
)

env = gym.make("morl/FrozenLakePlus-v1", render_mode="human", map_name=None, custom_map_size=20)

# @render_browser
def run():
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # yield env.render()
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        # print(reward)
        print(observation)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    run()