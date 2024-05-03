import numpy as np
from itertools import chain

class EpsilonGreedy():
    def __init__(self, epsilon, seed):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.eval = False

    def choose_action(self, action_space, q_values):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon and not self.eval:
            # print("SAMPLE")
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(q_values) == q_values[0]: action = action_space.sample()
            else: action = np.argmax(q_values)
        return action

    def update(self, **kwargs):
        return None
    
class MO_EpsilonGreedy(EpsilonGreedy):
    """ Choses e-greedy action based on passed in scalarization of Q_function."""
    def __init__(self, epsilon, seed, scalar_vector):
        super().__init__(epsilon, seed)
        self.scalar_vector = scalar_vector

    def scalarize(self, q_values, scalar_vector=None):
        # multiply the scalars in scalar vector with the vectors in q_values then return
        if scalar_vector is None:
            scalar_vector = self.scalar_vector
        #return np.dot(scalar_vector, q_values)
        return q_values[1]


    def choose_action(self, action_space, q_values, scalar_vector=None):
        if scalar_vector is None:
            scalar_vector = self.scalar_vector
        q_values = self.scalarize(q_values)
        return super().choose_action(action_space, q_values)

    def update(self, scalar_vector, **kwargs):
        self.scalar_vector = scalar_vector

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
            + self.gamma * np.max(self.action_values(new_state), axis=0)
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        self.qtable[state, action] = q_update

    def action_values(self, state):
        return self.qtable[state]
        
    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

    def set_qtable(self, qtables):
        self.qtable = qtables

    def get_qtable(self):
        return self.qtable

    def average_qtable(self, lists_of_qtables):
        return np.array(lists_of_qtables).mean(axis=0)

class LinearQlearning():
    def __init__(self, learning_rate, gamma, state_dim, action_size):
        self.feature_dim = state_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.reset_qtable() #stores feature weights 

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.action_values(new_state), axis=0)
            - self.q_value(state, action)
        )
        # print(delta)
        regularization_term = 0.1 * self.w[action]

        self.w[action] = self.w[action] + self.learning_rate * (delta * np.array(state) - regularization_term)
        # self.bias += self.learning_rate * delta

    def q_value(self, state, action):
        #print('state', state)
        try:
            result = np.dot(self.w[action], state) #(self.w[action] * state).sum() + self.bias
        except FloatingPointError as e:
            print("Overflow", state, self.w[action], self.bias)

        return result

    def action_values(self, state):
        #create state action vector for all actions
        # f_a = np.repeat(np.expand_dims(state, axis=0), self.action_size, axis=0)
        # return (self.w * f_a).sum(axis = 1) + self.bias
        return np.dot(self.w, state) + self.bias
        
    def reset_qtable(self):
        """Reset the Q-table."""
        self.w = np.zeros((self.action_size, self.feature_dim))
        self.bias = 0
        self.qtable = (self.w, self.bias)

    def set_qtable(self, qtables):
        self.qtable = qtables
        self.w, self.bias = qtables

    def get_qtable(self):
        return (self.w, self.bias)

    def average_qtable(self, lists_of_qtables):
        return np.array(lists_of_qtables).mean(axis=0)

class MO_LinearQlearning():
    """Q-learning formulation that tracks and updates multiple value functions"""
    def __init__(self, learning_rate, gamma, state_info, action_size, reward_dim):
        state_size, state_dim = state_info
        self.qfuncs = [Qlearning(learning_rate, gamma, state_size, action_size),
                       LinearQlearning(learning_rate, gamma, state_dim, action_size)]
        self.reward_dim = reward_dim

        self.reset_qtable()
        #np.seterr(all='raise')

    def update(self, state, action, reward, new_state):
        # print(reward)
        for ind, qfunc in enumerate(self.qfuncs):
            qfunc.update(state[ind], action, reward[ind], new_state[ind])

    def reset_qtable(self):
        """Reset the Q-table."""
        for qfunc in self.qfuncs: qfunc.reset_qtable()

    def action_values(self, state):
        return np.array([qfunc.action_values(state[ind]) for ind, qfunc in enumerate(self.qfuncs)])

    def set_qtable(self, qtables):
        for i, qtable in enumerate(qtables): self.qfuncs[i].set_qtable(qtable)

    def get_qtable(self):
        return [qfunc.qtable for qfunc in self.qfuncs]

    def average_qtable(self, lists_of_qtables):
        avg_qtables = []
        for i in range(self.reward_dim):
            avg_qtables.append(np.array([qtable[i] for qtable in lists_of_qtables]).mean(axis=0))
        return avg_qtables
    
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
        rs = state - self.state_low

        start_pos = rs + self.offsets
        tiles = np.floor(start_pos/self.tile_width).astype(int)
        return self.tile_values[tuple(tiles)].mean(axis=(0, 1))
    
    def update(self, state, action, reward, new_state):
        rs = state - self.state_low
        qa = self.action_values(state)[action]

        start_pos = rs + self.offsets
        tiles = np.floor(start_pos/self.tile_width).astype(int)

        delta = reward + self.gamma * np.max(self.action_values(new_state), axis=0) - qa
        self.tile_values[tuple(tiles)] += self.learning_rate * delta
        
    def reset_qtable(self):
        """Reset the Q-table."""
        tile_value_shape = chain(self.num_tilings, self.tile_dim, self.num_actions) 
        self.tile_values = np.zeros(list(tile_value_shape))

class MO_Qlearning():
    """Q-learning formulation that tracks and updates multiple value functions"""
    def __init__(self, learning_rate, gamma, state_size, action_size, reward_dim):
        self.qfuncs = [Qlearning(learning_rate, gamma, state_size[i], action_size) for i in range(reward_dim)]
        self.reward_dim = reward_dim

        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        for i in range(self.reward_dim):
            self.qfuncs[i].update(state[i], action, reward[i], new_state[i])

    def reset_qtable(self):
        """Reset the Q-table."""
        for qfunc in self.qfuncs:
            qfunc.reset_qtable()

    def action_values(self, state):
        return np.array([qfunc.action_values(state[qidx]) for qidx, qfunc in enumerate(self.qfuncs)])

    def set_qtable(self, qtables):
        for i, qtable in enumerate(qtables):
            self.qfuncs[i].qtable = qtable

    def get_qtable(self):
        return [qfunc.qtable for qfunc in self.qfuncs]

    def average_qtable(self, lists_of_qtables):
        avg_qtables = []
        for i in range(self.reward_dim):
            avg_qtables.append(np.array([qtable[i] for qtable in lists_of_qtables]).mean(axis=0))
        return avg_qtables


class MO_TC_Qlearning:
    """ MO-Q learning using tile coding as the function approximator """
    def __init__(self, learning_rate:float, gamma:float, state_low:np.array,
                 state_high:np.array, num_tilings:int, action_size:int,
                 tile_width:np.array, reward_dim):
        
        super().__init__(learning_rate, gamma, state_low, state_high, num_tilings, action_size, tile_width)

        self.reward_dim = np.zeros(reward_dim)
        self.reward_weights = np.ones(reward_dim)/(reward_dim)
        self.reset_qtable()

    def update_weights(self):
        return None
    
    def update(self, state, action, reward, new_state):
        self.update_weights()
        self.super().update(state, action, reward, new_state)

    def reset_qtable(self):
        """Reset the Q-table."""
        tile_value_shape = chain(self.num_tilings, self.tile_dim, self.num_actions, self.reward_dim) 
        self.tile_values = np.zeros(list(tile_value_shape))


        
