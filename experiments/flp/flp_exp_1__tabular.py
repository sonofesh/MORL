# Author: Andrea Pierré
# License: MIT License
# MOST IS TAKEN FROM https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/

from functools import partial

from pathlib import Path

import pandas as pd

from experiments.flp.params import Params
from experiments.flp.tabular_lifecycles import run_training, tabular_mo_state, tabular_state, vis_run
from experiments.flp.utils import get_flp_env
from models.scalar_morl import Qlearning, EpsilonGreedy, MO_EpsilonGreedy, MO_Qlearning



def baseline_reward_fn(reward, lambda1=1.0, lambda2=1.0):
    goal_achieved = lambda1 * reward[0]
    coin_collected = lambda2 * reward[1]
    return goal_achieved + coin_collected

def simple_morl_reward_fn(reward, lambda1=1.0, lambda2=1.0):
    goal_achieved = lambda1 * reward[0]
    coin_collected = lambda2 * reward[1]
    return goal_achieved, coin_collected




def run_experiment_1():
    params = Params(
        total_episodes=1000,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        map_size=5,
        seed=123,
        is_slippery=False,
        n_runs=20,
        action_size=None,
        state_size=None,
        proba_frozen=0.5,
        proba_coin=0.4,
        proba_hole=0.1,
        savefig_folder=Path("../../_static/img/tutorials/"),
    )

    # Create the figure folder if it doesn't exists
    params.savefig_folder.mkdir(parents=True, exist_ok=True)



    map_sizes = [4]#, 7, 9, 11]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:

        simple_morl_setup = lambda params: (
            MO_Qlearning(
                learning_rate=params.learning_rate,
                gamma=params.gamma,
                state_size=[params.map_size**2, params.state_size],
                action_size=params.action_size,
                reward_dim=2
            ), MO_EpsilonGreedy(
                epsilon=params.epsilon,
                seed=params.seed,
                scalar_vector=[1, 0]
            )
        )

        scalar_vector_update_schedule = [
            [500, [0.67, 0.33]],
            [1000, [0.34, 0.66]],
            [1500, [0.2, 0.8]],
            [1900, [0.5, 0.5]],
        ]


        # _, _, qtable = run_training(
        #     params,
        #     simple_morl_reward_fn,
        #     map_size=map_size,
        #     setup_learning_and_explorer_code=simple_morl_setup,
        #     scalar_vector_update_schedule=scalar_vector_update_schedule,
        #     tabular_state_fn=tabular_mo_state,
        #     eval_total_episodes=10,
        #     eval_frequency=250
        # )


        scalar_vector_update_schedule_inner_episode = [
            [0, [0.0, 1.0]],
            [5, [0.1, 0.9]],
            [10, [0.4, 0.6]],
            [15, [0.9, 0.1]],
            [20, [1.0, 0.0]],
        ]

        scalar_vector_update_schedule = [
            [0, [1.0, 0.0]],
            [500, scalar_vector_update_schedule_inner_episode]
        ]

        _, _, qtable = run_training(
            params,
            simple_morl_reward_fn,
            map_size=map_size,
            setup_learning_and_explorer_code=simple_morl_setup,
            scalar_vector_update_schedule=scalar_vector_update_schedule,
            # scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode,
            tabular_state_fn=tabular_mo_state,
            eval_total_episodes=10,
            eval_frequency=250
        )

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

        env = get_flp_env(params, map_size, render_mode="human")
        while True:
            vis_run(partial(setup_learning_and_explorer_code, qtable), tabular_mo_state, params, env, simple_morl_reward_fn, map_size=map_size, scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode)



    # plot_steps_and_rewards(res_all, st_all, params)


if __name__ == "__main__":
    run_experiment_1()