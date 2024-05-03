# Author: Andrea Pierré
# License: MIT License
# MOST IS TAKEN FROM https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
import json
from functools import partial
import sys

from pathlib import Path

from experiments.flp.params import Params
from experiments.flp.tabular_lifecycles import run_training, tabular_mo_state, tabular_state, vis_run, just_state, get_features
from experiments.flp.utils import get_flp_env
from models.scalar_morl import LinearQlearning, Qlearning, EpsilonGreedy, MO_EpsilonGreedy, MO_Qlearning, MO_LinearQlearning

def baseline_goal_only_reward_fn(reward, lambda1=1.0, lambda2=1.0):
    goal_achieved = lambda1 * reward[0]
    coin_collected = lambda2 * reward[1]
    return goal_achieved #+ coin_collected

def baseline_reward_fn(reward, lambda1=1.0, lambda2=1.0):
    goal_achieved = lambda1 * reward[0]
    coin_collected = lambda2 * reward[1]
    return goal_achieved + coin_collected #- 0.1

def simple_morl_reward_fn(reward, lambda1=1.0, lambda2=1.0):
    goal_achieved = lambda1 * reward[0]
    coin_collected = lambda2 * reward[1]
    return goal_achieved, coin_collected


def run_experiment_1():
    import pandas as pd

    params = Params(
        total_episodes=2000,
        max_episode_len=None,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.1,
        map_size=5,
        seed=123,
        state_dim=2,
        is_slippery=False,
        n_runs=20,
        action_size=None,
        state_size=None,
        proba_frozen=0.5,
        proba_coin=0.4,
        proba_hole=0.1,
        eval_total_episodes=10,
        eval_freq=250,
        savefig_folder=Path("../../_static/img/tutorials/"),
    )

    # Create the figure folder if it doesn't exists
    params.savefig_folder.mkdir(parents=True, exist_ok=True)

    SKIP_GO = True
    # 3.8 reward | 90% completion
    SKIP_BASE = True
    SKIP_SIMPLE_MORL = True
    SKIP_INTEREPISODE_MORL = False

    map_sizes = [4]#, 7, 9, 11]
    #map_sizes = [20]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:
        if not SKIP_GO:
            go_setup = lambda params: (
                Qlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_size=params.map_size**2,
                    action_size=params.action_size,
                ), EpsilonGreedy(
                    epsilon=params.epsilon,
                    seed=params.seed,
                )
            )
            _, _, qtable, baseline_go_res = run_training(
                params,
                baseline_goal_only_reward_fn,
                map_size=map_size,
                setup_learning_and_explorer_code=go_setup,
                tabular_state_fn=just_state,
                eval_total_episodes=params.eval_total_episodes,
                eval_frequency=params.eval_freq
            )

            json.dump(baseline_go_res, open("exp1_results/go_baseline.json", "w"))

        if not SKIP_BASE:
            baseline_setup = lambda params: (
                LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_dim=params.state_dim,
                    action_size=params.action_size,
                ), EpsilonGreedy(
                    epsilon=params.epsilon,
                    seed=params.seed,
                )
            )

            _, _, qtable, baseline_res = run_training(
                params,
                baseline_reward_fn,
                map_size=map_size,
                setup_learning_and_explorer_code=baseline_setup,
                tabular_state_fn=get_features,
                eval_total_episodes=params.eval_total_episodes,
                eval_frequency=params.eval_freq
            )

            json.dump(baseline_res, open("exp1_results/baseline.json", "w"))


            env = get_flp_env(params, map_size, render_mode="human")

            def load(qtable, params):
                learner = LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_dim=params.state_dim,
                    action_size=params.action_size,
                )
                exp = EpsilonGreedy(
                    epsilon=params.epsilon,
                    seed=params.seed,
                )
                learner.set_qtable(qtable)
                return learner, exp


            def setup_learning_and_explorer_code(qtable, params):
                learner = LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_dim=params.state_dim,
                    action_size=params.action_size,
                )
                explorer = EpsilonGreedy(
                    epsilon=params.epsilon,
                    seed=params.seed,
                )

                learner.set_qtable(qtable)
                return learner, explorer

            env = get_flp_env(params, map_size, render_mode="human")
            # while True:
            #     vis_run(partial(setup_learning_and_explorer_code, qtable), tabular_mo_state, params, env, simple_morl_reward_fn, map_size=map_size, scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode)

            # save qtable as list of numpy arrays
            # if isinstance(qtable, list):
            #     qtable = [qtable[0][i].tolist() for i in range(len(qtable))]
            # q_values = qtable[0]
            # w2,b2 = qtable[0]
            # print(q_values)
            # # qtable = [q_values.tolist(), [w2.tolist(), b2]]
            # qtable = [w2.tolist(), b2]
            # #
            # json.dump(qtable, open("../../models/checkpoints/demo_qtable.json", "w"))
            #
            # # load qtable from file
            # import numpy as np
            # qtables = json.load(open("../../models/checkpoints/demo_qtable.json"))
            # # qtable = [np.array(x) for x in qtables]
            # # qtable = [[np.array(x[0]), x[1]] for x in qtables]
            # qtable = [np.array(qtables[0]), [np.array(qtables[1][0]), qtables[1][1]]]

            while True:
                vis_run(partial(setup_learning_and_explorer_code, qtable), get_features, params, env, baseline_reward_fn, map_size=map_size)



            # while True:
            #     vis_run(partial(load, qtable), get_features, params, env, simple_morl_reward_fn, map_size=map_size)


        if not SKIP_SIMPLE_MORL:
            simple_morl_setup = lambda params: (
                MO_LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_info=[params.map_size**2, params.state_dim],
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


            _, _, qtable, simple_morl_res = run_training(
                params,
                simple_morl_reward_fn,
                map_size=map_size,
                setup_learning_and_explorer_code=simple_morl_setup,
                scalar_vector_update_schedule=scalar_vector_update_schedule,
                tabular_state_fn=tabular_mo_state,
                eval_total_episodes=params.eval_total_episodes,
                eval_frequency=params.eval_freq
            )
            json.dump(simple_morl_res, open("exp1_results/simple_morl.json", "w"))

        
        if not SKIP_INTEREPISODE_MORL:
            simple_morl_setup = lambda params: (
                MO_LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_info=[params.map_size**2, params.state_dim], #[params.map_size**2, params.state_size],
                    action_size=params.action_size,
                    reward_dim=2
                ), MO_EpsilonGreedy(
                    epsilon=params.epsilon,
                    seed=params.seed,
                    scalar_vector=[1, 0]
                )
            )

            scalar_vector_update_schedule_inner_episode = [
                [0, [0.00, 1.0]],
                [10, [0.1, 0.9]],
                [20, [0.5, 0.5]],
                [40, [1.0, 0.0]],
            ]

            scalar_vector_update_schedule = [
                # [0, [1.0, 0.0]],
                # [750, [0.1, 0.9]],
                [0, scalar_vector_update_schedule_inner_episode]
            ]

            _, _, qtable, inter_episode_scheduling_res = run_training(
                params,
                simple_morl_reward_fn,
                map_size=map_size,
                setup_learning_and_explorer_code=simple_morl_setup,
                scalar_vector_update_schedule=scalar_vector_update_schedule,
                # scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode,
                tabular_state_fn=tabular_mo_state,
                eval_total_episodes=params.eval_total_episodes,
                eval_frequency=params.eval_freq
            )
            json.dump(inter_episode_scheduling_res, open("exp1_results/morl_interepisode.json", "w"))


            def setup_learning_and_explorer_code(qtable, params):
                learner = MO_LinearQlearning(
                    learning_rate=params.learning_rate,
                    gamma=params.gamma,
                    state_info=[params.map_size**2, params.state_dim],
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
            # while True:
            #     vis_run(partial(setup_learning_and_explorer_code, qtable), tabular_mo_state, params, env, simple_morl_reward_fn, map_size=map_size, scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode)

            # save qtable as list of numpy arrays
            # if isinstance(qtable, list):
            #     qtable = [qtable[0][i].tolist() for i in range(len(qtable))]
            q_values = qtable[0]
            w2,b2 = qtable[1]
            # print(q_values)
            print(w2)
            qtable = [q_values.tolist(), [w2.tolist(), b2]]

            json.dump(qtable, open("../../models/checkpoints/demo_qtable.json", "w"))

            # load qtable from file
            import numpy as np
            qtables = json.load(open("../../models/checkpoints/demo_qtable.json"))
            # qtable = [np.array(x) for x in qtables]
            # qtable = [[np.array(x[0]), x[1]] for x in qtables]
            qtable = [np.array(qtables[0]), [np.array(qtables[1][0]), qtables[1][1]]]

            while True:
                vis_run(partial(setup_learning_and_explorer_code, qtable), tabular_mo_state, params, env, simple_morl_reward_fn, map_size=map_size, 
                        scalar_vector_update_schedule_inner_episode=scalar_vector_update_schedule_inner_episode)




    # plot_steps_and_rewards(res_all, st_all, params)


if __name__ == "__main__":
    run_experiment_1()