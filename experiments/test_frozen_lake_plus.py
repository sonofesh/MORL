import gymnasium as gym

from gymnasium.envs.registration import register

register(
    id='morl/FrozenLakePlus-v1',
    entry_point='environments.frozen_lake_plus.frozen_lake_plus:FrozenLakePlusEnv',
    max_episode_steps=300,
)

env = gym.make("morl/FrozenLakePlus-v1", render_mode="human", map_name=None, custom_map_size=20)

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   print(reward)

   if terminated or truncated:
      observation, info = env.reset()

env.close()