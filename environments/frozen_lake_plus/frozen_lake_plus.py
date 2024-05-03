import math
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import logging
logging.getLogger().setLevel(logging.CRITICAL)

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SCCF", "CHCH", "FCFH", "HCFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8, p: List[float] = [0.7, 0.2, 0.1], seed: Optional[int] = None,
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p[0] = min(1, p[0])
        board = np_random.choice(["F", "H", "C"], (size, size), p=p)
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


# get minimum of list based on absolute value
def get_abs_min(row, col, coin_dist):
    distances = abs(row - coin_dist[:, 0]) + abs(col - coin_dist[:, 1])

    min_ind = np.argmin(distances)
    return coin_dist[min_ind, 0] - row, coin_dist[min_ind, 1] - col

def normalize(val, min_val, max_val):
    # return val
    return (val - min_val)/(max_val - min_val)

def euc_dist(row, col, target, norm = 2):
    return row - target[0], col - target[1]
    distance = pow(row - target[0], norm) + pow(col - target[1], norm)
    return pow(distance, 1/norm)

class FrozenLakePlusEnv(Env):
    """
    Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
    by walking over the frozen lake.
    The player may not always move in the intended direction due to the slippery nature of the frozen lake.

    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

    Holes in the ice are distributed in set locations when using a pre-determined map
    or in random locations when a random map is generated.

    The player makes moves until they reach the goal or fall in a hole.

    The lake is slippery (unless disabled) so the player may move perpendicular
    to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).

    Randomly generated worlds will always have a path to the goal.

    Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    C
    p
    -> [-1, 0]

    p
    C
    -> [1, 0]

    C p
    -> [0, -1]

    p C
    -> [0, 1]

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * ncols + current_col (where both the row and col start at 0).

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state.

    See <a href="#is_slippy">`is_slippery`</a> for transition probability information.


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc=None`: Used to specify maps non-preloaded maps.

    Specify a custom map.
    ```
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    ```

    A random generated map can be specified by calling the function `generate_random_map`.
    ```
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    ```

    `map_name="4x4"`: ID to use any of the preloaded maps.
    ```
        "4x4":[
            "SCFF",
            "CHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
    ```

    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
    `None` a random 8x8 map with 80% of locations frozen will be generated.

    <a id="is_slippy"></a>`is_slippery=True`: If true the player will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.

    For example, if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3


    ## Version History
    * v1: Bug fixes to rewards
    * v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        reset_coins=True,
        custom_map_size=8,
        use_coin_dist: bool = True,
        use_goal_dist_feat: bool = False,
        use_ice_feat: bool = False,
        use_ice_existence_feat: bool = True,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map(size=custom_map_size)
        elif desc is None:
            desc = MAPS[map_name]

        self.use_coin_dist = use_coin_dist
        self.desc = desc = np.asarray(desc, dtype="c")
        self.orig_desc = desc.copy()

        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.is_slippery = is_slippery
        self.reset_coins = reset_coins

        self.p_state = 0

        self.use_goal_dist_feat = use_goal_dist_feat
        self.use_ice_feat = use_ice_feat
        self.use_ice_existence_feat = use_ice_existence_feat

        self.features_dim = 0
        if self.use_coin_dist:
            self.features_dim += 2
        if self.use_goal_dist_feat:
            self.features_dim += 2
        if self.use_ice_feat:
            self.features_dim += 2
        if self.use_ice_existence_feat:
            self.features_dim += 4


        self.coin_coords = []

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.nA = nA
        self.nS = nS

        self.set_p()

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.coin_img = None

    def set_p(self):
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.coin_coords = []
        self.ice_coords = []
        self.goal = ()

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"C":
                        self.coin_coords.append((row, col))
                    if letter in b"GH":
                        if letter == b"G": self.goal = (row, col)
                        if letter == b"H": self.ice_coords.append((row, col))
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *self.update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *self.update_probability_matrix(row, col, a)))
                    self.P[s][a] = li

        self.coin_coords = np.array(self.coin_coords)
        self.ice_coords = np.array(self.ice_coords)

    def from_s(self, s):
        return s // self.ncol, s % self.ncol

    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return (row, col)


    def update_probability_matrix(self, row, col, action):
        newrow, newcol = self.inc(row, col, action)
        newstate = self.to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"GH"
        # reward = float(newletter == b"G")
        # reward += reward + float(newletter == b"C")
        reward = [0, 0]
        if newletter == b"G":
            reward = [1, 0]
        elif newletter == b"C":
            reward = [0, 1]

        return newstate, reward, terminated

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.desc[self.from_s(s)] == b"C":
            self.desc[self.from_s(s)] = b"F"
            self.set_p()

        #if self.desc[self.from_s(s)] == b"H": print('fail')

        row, col = self.from_s(s)
        if len(self.coin_coords) > 0: coin_dist = get_abs_min(row, col, self.coin_coords)
        else: coin_dist = (0, 0)

        if len(self.ice_coords) > 0: ice_dist = get_abs_min(row, col, self.ice_coords)
        else: ice_dist = (0, 0)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        info = {"prob": p, "at_goal": self.desc[self.from_s(s)] == b"G", 'coins_cleared':  len(self.coin_coords) == 0 and self.use_coin_dist}
        # if t and self.desc[self.from_s(s)] != b"G":
        #     r = [x-0.1 for x in r]

        if self.features_dim > 0:
            #print('max_dist', euc_dist(0, 0, self.goal))
            # goal_dist = normalize(self.goal[0] - row, 0, self.nrow), normalize(self.goal[1] - col, col, self.goal)[1], 0, self.ncol)
            goal_dist = normalize(self.goal[0] - row, 0, self.nrow), normalize(self.goal[1] - col, 0, self.ncol)
            # coin_dist = normalize(coin_dist[0], -self.nrow, self.nrow), normalize(coin_dist[1], -self.ncol, self.ncol)
            # ice_dist = normalize(ice_dist[0], -self.nrow, self.nrow), normalize(ice_dist[1], -self.ncol, self.ncol)
            coin_dist = normalize(coin_dist[0], 0, self.nrow), normalize(coin_dist[1], -self.ncol, self.ncol)
            ice_dist = normalize(ice_dist[0], 0, self.nrow), normalize(ice_dist[1], -self.ncol, self.ncol)

            exsist_ice_vals = [
                int(any([all(x == self.inc(*self.from_s(self.s), LEFT)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), UP)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), DOWN)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), RIGHT)) for x in self.ice_coords]))
            ]

            features = []
            if self.use_coin_dist:
                features.extend(coin_dist)
            if self.use_goal_dist_feat:
                features.extend(goal_dist)
            if self.use_ice_feat:
                features.extend(ice_dist)
            if self.use_ice_existence_feat:
                features.extend(exsist_ice_vals)


            # features = coin_dist + goal_dist[0] + ice_dist

            # if self.p_state is not None and self.p_state == s:
            #     r = [x-0.1 for x in r]
            self.p_state = s

            # print(s, features)
            #return a int state rep and then a tuple of state features
            return [int(s), features], r, t, False, info
        return int(s), r, t, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.p_state = 0
        super().reset(seed=seed)
        if self.reset_coins:
            self.desc = self.orig_desc.copy()
            self.set_p()

        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()

        row, col = self.from_s(self.s)
        if len(self.coin_coords) > 0:
            coin_dist = get_abs_min(row, col, self.coin_coords)
        else: coin_dist = (20,20)

        if len(self.ice_coords) > 0: ice_dist = get_abs_min(row, col, self.ice_coords)
        else: ice_dist = (20, 20)

        if self.features_dim > 0:
            #print('max_dist', euc_dist(0, 0, self.goal))
            # goal_dist = normalize(euc_dist(row, col, self.goal), 0, euc_dist(0, 0, self.goal)),
            # goal_dist = normalize(euc_dist(row, col, self.goal)[0], 0, euc_dist(0, 0, self.goal)[0]),
            # goal_dist = normalize(euc_dist(row, col, self.goal)[0], 0, euc_dist(0, 0, self.goal)[0]), normalize(euc_dist(row, col, self.goal)[1], 0, euc_dist(0, 0, self.goal)[1]),
            goal_dist = normalize(self.goal[0] - row, 0, self.nrow), normalize(self.goal[1] - col, 0, self.ncol)

            # coin_dist = normalize(coin_dist[0], -self.nrow, self.nrow), normalize(coin_dist[1], -self.ncol, self.ncol)
            # ice_dist = normalize(ice_dist[0], -self.nrow, self.nrow), normalize(ice_dist[1], -self.ncol, self.ncol)
            coin_dist = normalize(coin_dist[0], 0, self.nrow), normalize(coin_dist[1], -self.ncol, self.ncol)
            ice_dist = normalize(ice_dist[0], 0, self.nrow), normalize(ice_dist[1], -self.ncol, self.ncol)

            exsist_ice_vals = [
                int(any([all(x == self.inc(*self.from_s(self.s), LEFT)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), UP)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), DOWN)) for x in self.ice_coords])),
                int(any([all(x == self.inc(*self.from_s(self.s), RIGHT)) for x in self.ice_coords]))
            ]

            # features = coin_dist + goal_dist[0] + ice_dist
            #return a int state rep and then a tuple of state features
            features = []
            if self.use_coin_dist:
                features.extend(coin_dist)
            if self.use_goal_dist_feat:
                features.extend(goal_dist)
            if self.use_ice_feat:
                features.extend(ice_dist)
            if self.use_ice_existence_feat:
                features.extend(exsist_ice_vals)


            return [int(self.s), features], {"prob": 1}
        
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:

            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.coin_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.coin_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)
                elif desc[y][x] == b"C":
                    self.window_surface.blit(self.coin_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        elif desc[bot_row][bot_col] == b"C":
            self.window_surface.blit(self.ice_img, cell_rect)
            self.window_surface.blit(elf_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/