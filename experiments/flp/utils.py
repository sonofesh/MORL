import numpy as np
from gymnasium.envs.registration import register
import gymnasium as gym
from environments.frozen_lake_plus.frozen_lake_plus import generate_random_map






def postprocess(episodes, params, rewards, steps, map_size):
    import pandas as pd

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
    import matplotlib.pyplot as plt

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
    import seaborn as sns
    sns.set_theme()
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
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



def run_checkpoint_in_website(checkpoint):
    from models.scalar_morl import MO_Qlearning, MO_EpsilonGreedy
    import numpy as np
    import json
    from experiments.flp.tabular_lifecycles import vis_run, tabular_mo_state
    from experiments.flp.flp_exp_1__tabular import simple_morl_reward_fn
    from functools import partial
    from experiments.flp.params import Params
    from pathlib import Path

    def setup_learning_and_explorer_code(qtable, params):
        learner = MO_Qlearning(
            learning_rate=params.learning_rate,
            gamma=params.gamma,
            state_size=[params.map_size ** 2, params.state_size],
            action_size=params.action_size,
            reward_dim=2
        )
        explorer = MO_EpsilonGreedy(
            epsilon=params.epsilon,
            seed=params.seed,
            scalar_vector=[1, 1]
        )

        learner.set_qtable(qtable)
        return learner, explorer

    params = Params(
        total_episodes=2000,
        max_episode_len=None,
        learning_rate=0.4,
        gamma=0.99,
        epsilon=0.1,
        map_size=4,
        seed=123,
        is_slippery=False,
        n_runs=20,
        action_size=None,
        state_size=None,
        proba_frozen=0.5,
        proba_coin=0.4,
        proba_hole=0.1,
        eval_total_episodes=10,
        eval_freq=100,
        savefig_folder=Path("../../_static/img/tutorials/"),
    )


    env = get_flp_env(params, params.map_size, render_mode="human")
    # while True:
    #     vis_run(partial(setup_learning_and_explorer_code, qtable), tabular_mo_state, params, env, simple_morl_reward_fn, map_size=map_size, scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode)


    # load qtable from file
    qtables = json.load(open("qtable.json"))
    qtable = [np.array(x) for x in qtables]

    while True:
        scalar_vector_update_schedule_inner_episode = scalar_vector_update_schedule_inner_episode.copy() if scalar_vector_update_schedule_inner_episode else None
        if not map_size:
            map_size = params.map_size
        params = params._replace(action_size=env.action_space.n)
        params = params._replace(state_size=env.observation_space.n * map_size * 2)
        params = params._replace(map_size=map_size)

        learner, explorer = setup_learning_and_explorer_code(params)
        explorer.eval = True

        state = tabular_mo_state(env.reset(seed=params.seed)[0], map_size=params.map_size)  # Reset the environment

        step = 0
        done = False
        total_rewards = 0

        while not done:
            # uncomment in the index.html code
            #speed = document.querySelector('#speed').value
            #await sleep(1 / int(speed))
            # goal_scale = document.querySelector('#gqfunc').value / 100
            # cookie_scale = document.querySelector('#cookiesqfunc').value / 100
            # scalar_vector = [goal_scale, cookie_scale]
            # explorer.update(scalar_vector=scalar_vector)

            action = explorer.choose_action(
                action_space=env.action_space, q_values=learner.action_values(state)
            )

            # Log all states and actions

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, raw_reward, terminated, truncated, info = env.step(action)
            new_state = tabular_mo_state(new_state, map_size=params.map_size)

            reward = simple_morl_reward_fn(raw_reward)

            done = terminated or truncated

            total_rewards += sum(reward) if isinstance(reward, list) or isinstance(reward, tuple) else reward
            step += 1

            # Our new state is state
            state = new_state
        explorer.eval = False