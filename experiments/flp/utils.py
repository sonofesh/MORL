import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register
import gymnasium as gym
from environments.frozen_lake_plus.frozen_lake_plus import generate_random_map

import seaborn as sns
sns.set_theme()




def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st




def plot_first_and_last_frames(env, map_size, first_frame, params, img_title_postfix=""):
    """Plot the last frame of the simulation and the policy learned."""
    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    ax[0].imshow(first_frame)
    ax[0].axis("off")
    ax[0].set_title("First frame")

    ax[1].imshow(env.render())
    ax[1].axis("off")
    ax[1].set_title("Last frame")

    img_title = f"frozenlake_q_values_{map_size}x{map_size}_{img_title_postfix}.png"
    # set plot title to img_title
    fig.suptitle(img_title)
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()



def plot_states_actions_distribution(states, actions, map_size, params):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_steps_and_rewards(rewards_df, steps_df, params):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def register_flp():
    register(
        id='morl/FrozenLakePlus-v1',
        entry_point='environments.frozen_lake_plus.frozen_lake_plus:FrozenLakePlusEnv',
        max_episode_steps=300,
    )

def get_flp_env(params, map_size=None, render_mode="rgb_array"):
    register_flp()

    return gym.make(
        "morl/FrozenLakePlus-v1",
        render_mode=render_mode,
        is_slippery=params.is_slippery,
        reset_coins=True,
        max_episode_steps=params.max_episode_len,
        desc=generate_random_map(
            size=map_size, p=[params.proba_frozen, params.proba_hole, params.proba_coin], seed=params.seed
        ),
    )