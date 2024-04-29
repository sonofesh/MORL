import numpy as np
from tqdm import tqdm
from experiments.flp.utils import get_flp_env, postprocess, plot_states_actions_distribution, plot_first_and_last_frames
from models.scalar_morl import Qlearning, EpsilonGreedy
import pandas as pd

def just_state(state, map_size):
    s, c_dist = state
    return s

def tabular_state(state, map_size):
    s, c_dist = state
    return (s * (map_size*2)) + c_dist

def tabular_mo_state(state, map_size):
    s, c_dist = state
    return s, (s * (map_size*2)) + c_dist


def run_env_fully_tabular(
        params,
        env,
        learner,
        explorer,
        reward_fn,
        total_episodes: int = None,
        n_runs: int = None,
        progress_bar: bool = True,
        training: bool = True,
        qtables: np.array = None,
        tabular_state_fn=tabular_state,
        scalar_vector_update_schedule_inner_episode=None
):
    scalar_vector_update_schedule_inner_episode = scalar_vector_update_schedule_inner_episode.copy() if scalar_vector_update_schedule_inner_episode else None

    if total_episodes is None:
        total_episodes = params.total_episodes
    if n_runs is None:
        n_runs = params.n_runs
    if not training and n_runs != 1:
        raise ValueError("When not training, n_runs must be 1")

    rewards = np.zeros((total_episodes, n_runs))
    steps = np.zeros((total_episodes, n_runs))
    episodes = np.arange(total_episodes)

    load_checkpoint = False
    if qtables is None:
        if training:
            qtables = []
        else:
            qtables = None
    else:
        load_checkpoint = True


    all_states = []
    all_actions = []
    stats = {
        'total_episodes': 0,
        'completed_episodes': 0,
        'cleared_coin_episodes': 0,
        'perfect_episodes': 0,
        'cummulative_rewards': []
    }

    for run in range(n_runs):  # Run several times to account for stochasticity
        if training:
            learner.reset_qtable()  # Reset the Q-table between runs
        if load_checkpoint:
            learner.set_qtable(qtables[run])

        for episode in tqdm(
            episodes, desc=f"Run {run}/{n_runs} - Episodes", leave=False, disable=not progress_bar
        ):
            state = tabular_state_fn(env.reset(seed=params.seed)[0], map_size=params.map_size)  # Reset the environment

            step = 0
            done = False
            total_rewards = 0
            goal_reached = False
            cleared_coins = False

            while not done:
                if scalar_vector_update_schedule_inner_episode and len(scalar_vector_update_schedule_inner_episode) > 0:
                    next_update = scalar_vector_update_schedule_inner_episode[0]
                    at_i = next_update[0]
                    if at_i == step:
                        scalar_vector = next_update[1]
                        explorer.update(scalar_vector=scalar_vector)
                        scalar_vector_update_schedule_inner_episode.pop(0)

                action = explorer.choose_action(
                    action_space=env.action_space, q_values=learner.action_values(state)
                )



                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, raw_reward, terminated, truncated, info = env.step(action)
                new_state = tabular_state_fn(new_state, map_size=params.map_size)

                reward = reward_fn(raw_reward)

                done = terminated or truncated
                if done:
                    goal_reached = info.get('at_goal')
                    cleared_coins = info.get('coins_cleared')

                if training:
                    learner.update(state, action, reward, new_state)

                total_rewards += sum(reward) if isinstance(reward, list) or isinstance(reward, tuple) else reward
                step += 1

                # Our new state is state
                state = new_state

            stats['total_episodes'] += 1

            if goal_reached:
                stats['completed_episodes'] += 1
            if cleared_coins:
                stats['cleared_coin_episodes'] += 1
            if goal_reached and cleared_coins:
                stats['perfect_episodes'] += 1

            stats['cummulative_rewards'].append(total_rewards)

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        if training:
            if load_checkpoint:
                qtables[run] = learner.get_qtable()
            else:
                qtables.append(learner.get_qtable())

    return rewards, steps, episodes, qtables, all_states, all_actions, stats


def run_training(
        params,
        reward_fn,
        setup_learning_and_explorer_code,
        map_size=None,
        eval_frequency: int = 100,
        eval_total_episodes: int = 20,
        scalar_vector_update_schedule=None,
        scalar_vector_update_schedule_inner_episode=None,
        tabular_state_fn=tabular_state
):
    scalar_vector_update_schedule = scalar_vector_update_schedule.copy() if scalar_vector_update_schedule else None
    scalar_vector_update_schedule_inner_episode = scalar_vector_update_schedule_inner_episode.copy() if scalar_vector_update_schedule_inner_episode else None

    if not map_size:
        map_size = params.map_size

    env = get_flp_env(params, map_size)

    env.reset()
    first_frame = env.render()

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n * map_size * 2)
    params = params._replace(map_size=map_size)


    env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space

    learner, explorer = setup_learning_and_explorer_code(params)

    print(f"Map size: {map_size}x{map_size}")
    total_training_episodes = params.total_episodes

    training_res_all = pd.DataFrame()
    training_st_all = pd.DataFrame()
    eval_res_all = pd.DataFrame()
    eval_st_all = pd.DataFrame()


    qtable = None
    qtables = None

    eval_recorded_stats = {}

    curr_scalar_vector_update_schedule_inner_episode = scalar_vector_update_schedule_inner_episode
    for i in range(total_training_episodes):
        if scalar_vector_update_schedule and len(scalar_vector_update_schedule) > 0:
            next_update = scalar_vector_update_schedule[0]
            at_i = next_update[0]

            if at_i == i:

                scalar_vector = next_update[1]
                if isinstance(scalar_vector[0], list):
                    curr_scalar_vector_update_schedule_inner_episode = scalar_vector
                else:
                    explorer.update(scalar_vector=scalar_vector)
                scalar_vector_update_schedule.pop(0)


        if i % eval_frequency == 0 and i > 0:
            explorer.eval = True
            all_eval_stats = []
            for qtbls in qtables:
                eval_rewards, eval_steps, eval_episodes, _, eval_all_states, eval_all_actions, eval_stats = run_env_fully_tabular(
                    params, env, learner, explorer, reward_fn, total_episodes=eval_total_episodes, n_runs=1, progress_bar=True, training=False, qtables=[qtbls], tabular_state_fn=tabular_state_fn, scalar_vector_update_schedule_inner_episode=curr_scalar_vector_update_schedule_inner_episode
                )
                all_eval_stats.append({
                    'cummulative_rewards': np.array(eval_stats['cummulative_rewards']).mean(),
                    'completed_episodes': eval_stats['completed_episodes'],
                    'cleared_coin_episodes': eval_stats['cleared_coin_episodes'],
                    'perfect_episodes': eval_stats['perfect_episodes'],
                    'total_episodes': eval_stats['total_episodes']
                })

            explorer.eval = False

            # Average everything in all_eval_stats
            eval_stats = {}
            for k in all_eval_stats[0].keys():
                eval_stats[k] = np.array([d[k] for d in all_eval_stats]).mean()

            eval_recorded_stats[str(i)] = eval_stats

            # eval_rewards, eval_steps, eval_episodes, _, eval_all_states, eval_all_actions, eval_stats = run_env_fully_tabular(
            #     params, env, learner, explorer, reward_fn, total_episodes=eval_total_episodes, n_runs=1, progress_bar=True, training=False, qtables=[qtable], tabular_state_fn=tabular_state_fn, scalar_vector_update_schedule_inner_episode=curr_scalar_vector_update_schedule_inner_episode
            # )

            # eval_stats['cummulative_rewards'] =  np.array(eval_stats['cummulative_rewards']).mean()

            explorer.eval = False
            plot_first_and_last_frames(env, map_size, first_frame, params)

            print("Training episode: ", i)
            print('Eval stats:', eval_stats)

        training_rewards, training_steps, training_episodes, qtables, training_all_states, training_all_actions, training_stats = run_env_fully_tabular(
            params, env, learner, explorer, reward_fn, qtables=qtables, total_episodes=1, tabular_state_fn=tabular_state_fn, scalar_vector_update_schedule_inner_episode=curr_scalar_vector_update_schedule_inner_episode
        )

        # Save the results in dataframes
        res, st = postprocess(training_episodes, params, training_rewards, training_steps, map_size)
        # qtable = qtables.mean(axis=0)  # Average the Q-table between runs
        #qtable = learner.average_qtable(qtables)
        # learner.set_qtable(qtable)

        training_res_all = pd.concat([training_res_all, res])
        training_st_all = pd.concat([training_st_all, st])

    explorer.eval = True
    all_eval_stats = []
    for qtbls in qtables:
        eval_rewards, eval_steps, eval_episodes, _, eval_all_states, eval_all_actions, eval_stats = run_env_fully_tabular(
            params, env, learner, explorer, reward_fn, total_episodes=eval_total_episodes, n_runs=1, progress_bar=True,
            training=False, qtables=[qtbls], tabular_state_fn=tabular_state_fn,
            scalar_vector_update_schedule_inner_episode=curr_scalar_vector_update_schedule_inner_episode
        )
        all_eval_stats.append({
            'cummulative_rewards': np.array(eval_stats['cummulative_rewards']).mean(),
            'completed_episodes': eval_stats['completed_episodes'],
            'cleared_coin_episodes': eval_stats['cleared_coin_episodes'],
            'perfect_episodes': eval_stats['perfect_episodes'],
            'total_episodes': eval_stats['total_episodes']
        })

    explorer.eval = False

    # Average everything in all_eval_stats
    eval_stats = {}
    for k in all_eval_stats[0].keys():
        eval_stats[k] = np.array([d[k] for d in all_eval_stats]).mean()

    eval_recorded_stats[str(i)] = eval_stats

    eval_stats['cummulative_rewards'] = np.array(eval_stats['cummulative_rewards']).mean()
    print("End of training")
    print('Eval stats:', eval_stats)

    eval_recorded_stats['EOT'] = eval_stats

    env.close()


    return training_res_all, training_st_all, qtables[0], eval_recorded_stats


def vis_run(setup_learning_and_explorer_code, tabular_state_fn, params, env, reward_fn, map_size=None, scalar_vector_update_schedule_inner_episode=None):
    scalar_vector_update_schedule_inner_episode = scalar_vector_update_schedule_inner_episode.copy() if scalar_vector_update_schedule_inner_episode else None
    if not map_size:
        map_size = params.map_size
    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n * map_size * 2)
    params = params._replace(map_size=map_size)


    learner, explorer = setup_learning_and_explorer_code(params)
    explorer.eval = True

    state = tabular_state_fn(env.reset(seed=params.seed)[0], map_size=params.map_size)  # Reset the environment

    step = 0
    done = False
    total_rewards = 0

    while not done:
        if scalar_vector_update_schedule_inner_episode and len(scalar_vector_update_schedule_inner_episode) > 0:
            next_update = scalar_vector_update_schedule_inner_episode[0]
            at_i = next_update[0]
            if at_i <= step:
                scalar_vector = next_update[1]
                # print(f"UPDATE TO {scalar_vector}")

                explorer.update(scalar_vector=scalar_vector)
                scalar_vector_update_schedule_inner_episode.pop(0)

        action = explorer.choose_action(
            action_space=env.action_space, q_values=learner.action_values(state)
        )

        # Log all states and actions

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, raw_reward, terminated, truncated, info = env.step(action)
        new_state = tabular_state_fn(new_state, map_size=params.map_size)

        reward = reward_fn(raw_reward)

        done = terminated or truncated

        total_rewards += sum(reward) if isinstance(reward, list) or isinstance(reward, tuple) else reward
        step += 1

        # Our new state is state
        state = new_state
    explorer.eval = False
    return