from pathlib import Path
from typing import NamedTuple

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    max_episode_len: int # number of possible steps in episode
    proba_frozen: float  # Probability that a tile is frozen
    proba_hole: float # Probability that a tile is a hole
    proba_coin: float # Probability that a tile is a coin (Coins are also FROZEN)
    savefig_folder: Path  # Root folder where plots are saved
    eval_total_episodes: int
    eval_freq: int