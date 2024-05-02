import gymnasium as gym
from gymnasium.envs.registration import register
#from render.render import render_browser
from experiments.flp.utils import get_flp_env
from experiments.flp.params import Params
from pathlib import Path

params = Params(
    total_episodes=2000,
    max_episode_len=None,
    learning_rate=0.05,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    state_dim=6,
    proba_frozen=0.5,
    proba_coin=0.4,
    proba_hole=0.1,
    eval_total_episodes=10,
    eval_freq=100,
    savefig_folder=Path("../../_static/img/tutorials/"),
)

env = get_flp_env(params, 4, render_mode='human')

# @render_browser
def run():
    observation, info = env.reset(seed=123)
    for _ in range(1000):
        # yield env.render()
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        # print(reward)
        # print(observation)

        if terminated or truncated:
            observation, info = env.reset(seed=123)

    env.close()

if __name__ == "__main__":
    run()