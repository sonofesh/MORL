import numpy as np
from itertools import chain

class EpsilonGreedy():
    def __init__(self, epsilon, seed):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose_action(self, action_space, q_values):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon: action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(q_values) == q_values[0]: action = action_space.sample()
            else: action = np.argmax(q_values)
        return action
    
class MO_EpsilonGreedy(EpsilonGreedy):
    """ Choses e-greedy action based on passed in scalarization of Q_function."""
    def __init__(self, epsilon, seed):
        super().__init__(epsilon, seed)

    def scalarize(self, q_values, scalar_vector):
        return np.sum(np.expand_dims(scalar_vector, axis = -1) * q_values, axis = 0)

    def choose_action(self, action_space, q_values, scalar_vector):
        q_values = self.scalarize(q_values)
        return super().choose_action(action_space, q_values)

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :], axis=0)
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        self.qtable[state, action] = q_update

    def action_values(self, state):
        return self.qtable[state]
        
    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

class TC_Qlearning():
    def __init__(self, learning_rate:float, gamma:float, state_low:np.array,
                 state_high:np.array, num_tilings:int, action_size:int,
                 tile_width:np.array):
        
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = action_size
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.learning_rate = learning_rate
        self.gamma = gamma

        dim = len(tile_width)
        self.tile_dim = np.ceil((state_high - state_low)/tile_width).astype(int) + 1

        tile_index = np.full((dim, num_tilings), range(num_tilings)).T
        self.offsets = (tile_index/num_tilings) * tile_width
        self.reset_qtable()

    def action_values(self, state):
        value = 0.
        rs = state - self.state_low

        start_pos = rs + self.offsets
        tiles = np.floor(start_pos/self.tile_width).astype(int)
        return self.tile_values[tuple(tiles)].mean()
    
    def update(self, state, action, reward, new_state):
        rs = state - self.state_low
        qa = self.action_values(state)[action]

        start_pos = rs + self.offsets
        tiles = np.floor(start_pos/self.tile_width).astype(int)

        delta = reward + self.gamma * np.max(self.action_values(new_state), axis=0) - qa
        q_update = qa + self.learning_rate * delta
        self.tile_values[tuple(tiles)] = q_update

    def action_values(self, state):
        return self.qtable[state]
        
    def reset_qtable(self):
        """Reset the Q-table."""
        tile_value_shape = chain(self.num_tilings, self.tile_dim, self.num_actions) 
        self.tile_values = np.zeros(list(tile_value_shape))

class MO_Qlearning(Qlearning):
    """
        Q-learning formulation that tracks and updates multiple value functions
    """
    def __init__(self, learning_rate, gamma, state_size, action_size, reward_dim):
        super().__init__(learning_rate, gamma, state_size, action_size)

        self.reward_dim = np.zeros(reward_dim)
        self.reward_weights = np.ones(reward_dim)/(reward_dim)
        self.reset_qtable()

    def update_weights():
        return None
    
    def update(self, state, action, reward, new_state):
        self.update_weights()
        self.super().update(state, action, reward, new_state)

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size, self.reward_dim))


class MO_TC_Qlearning:
    """ MO-Q learning using tile coding as the function approximator """
    def __init__(self, learning_rate:float, gamma:float, state_low:np.array,
                 state_high:np.array, num_tilings:int, action_size:int,
                 tile_width:np.array, reward_dim):
        
        super().__init__(learning_rate, gamma, state_low, state_high, num_tilings, action_size, tile_width)

        self.reward_dim = np.zeros(reward_dim)
        self.reward_weights = np.ones(reward_dim)/(reward_dim)
        self.reset_qtable()

    def update_weights():
        return None
    
    def update(self, state, action, reward, new_state):
        self.update_weights()
        self.super().update(state, action, reward, new_state)

    def reset_qtable(self):
        """Reset the Q-table."""
        tile_value_shape = chain(self.num_tilings, self.tile_dim, self.num_actions, self.reward_dim) 
        self.tile_values = np.zeros(list(tile_value_shape))


        
