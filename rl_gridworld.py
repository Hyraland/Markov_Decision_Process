import numpy as np 

class GridWorld(object):
    def __init__(self, rows, columns, start_p):
        self.rows =rows
        self.columns = columns
        self.i = start_p[0]
        self.j = start_p[1]
        self.grid = np.zeros((self.rows, self.columns))
        self.rewards = {}
        self.actions = {}

    def set(self, rewards, actions):
        self.rewards = rewards
        # since there're four possibilities of actions in total, we use integers 0-15 to represent possible actions. 
        # i.e. actions = 14 = bin(1110), meaning can choose up, down, and left. 
        # (four digits in binary represent up, down, left, right, respectively)
        self.actions = actions

    def move(self, action):
        # 0001 for right, 0010 for left, 0100 for down, 1000 for up
        if action == 1: self.j += 1
        elif action == 2: self.j -= 1
        elif action == 4: self.i += 1
        elif action == 8: self.i -= 1
        else:
            raise Exception("Invalid movement! action must be 1,2,4,8 but received", action)
        return self.reward()

    def get_next_state(self, s, a):
        i, j = s[0], s[1]
        if a & self.actions[(i,j)]:
            if a == 1: j += 1
            elif a == 2: j -= 1
            elif a == 4: i += 1
            elif a == 8: i -= 1
        return i, j


    def reward(self):
        return self.rewards.get((self.i,self.j),0)

    def current_state(self):
        return (self.i, self.j)

    def actions(self):
        if (self.i,self.j) in self.actions:
            return self.actions[(self.i,self.j)]
        else:
            return None

    def game_over(self):
        return self.actions != None

    def all_states(self):
	    # possibly buggy but simple way to get all states
	    # either a position that has possible next actions
	    # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())

def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    # 0001 for right, 0010 for left, 0100 for down, 1000 for up
    g = GridWorld(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
    (0, 0): (1,4),
    (0, 1): (1,2),
    (0, 2): (1,2,4),
    (1, 0): (4,8),
    (1, 2): (1,4,8),
    (2, 0): (1,8),
    (2, 1): (1,2),
    (2, 2): (1,2,8),
    (2, 3): (2,8),
    }
    g.set(rewards, actions)
    return g

class WindyGridWorld(object):
    def __init__(self, rows, columns, start_p):
        self.rows =rows
        self.columns = columns
        self.i = start_p[0]
        self.j = start_p[1]
        self.grid = np.zeros((self.rows, self.columns))
        self.rewards = {}
        self.actions = {}
        self.probs = {}

    def set(self, rewards, actions, probs):
        self.rewards = rewards
        # since there're four possibilities of actions in total, we use integers 0-15 to represent possible actions. 
        # i.e. actions = 14 = bin(1110), meaning can choose up, down, and left. 
        # (four digits in binary represent up, down, left, right, respectively)
        self.actions = actions
        self.probs = probs

    def move(self, action):
        # 0001 for right, 0010 for left, 0100 for down, 1000 for up
        s = self.current_state
        next_states_probs = self.probs[(s, action)]
        next_states = list(next_probs.keys)
        next_probs = list(next_probs.values)

        s2 = np.random.choice(next_states, p = next_probs)
        self.i, self.j = s2
        return self.rewards.get(s2, 0)

    def reward(self):
        if (self.i,self.j) in self.rewards:
            return self.rewards[(self.i,self.j)]
        else:
            return 0

    def current_state(self):
        return (self.i, self.j)

    def actions(self):
        if (self.i,self.j) in self.actions:
            return self.actions[(self.i,self.j)]
        else:
            return None

    def game_over(self):
        return self.actions != None

    def all_states(self):
	    # possibly buggy but simple way to get all states
	    # either a position that has possible next actions
	    # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())

def windy_grid():
    g = WindyGridWorld(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    # 1(0001) for Right, 2(0010) for Left, 4(0100) for Down, 8(1000) for Up
    actions = {
    (0, 0): (1,4),
    (0, 1): (1,2),
    (0, 2): (1,2,4),
    (1, 0): (4,8),
    (1, 2): (1,4,8),
    (2, 0): (1,8),
    (2, 1): (1,2),
    (2, 2): (1,2,8),
    (2, 3): (2,8),
    }

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
    ((2, 0), 8): {(1, 0): 1.0},
    ((2, 0), 4): {(2, 0): 1.0},
    ((2, 0), 2): {(2, 0): 1.0},
    ((2, 0), 1): {(2, 1): 1.0},
    ((1, 0), 8): {(0, 0): 1.0},
    ((1, 0), 4): {(2, 0): 1.0},
    ((1, 0), 2): {(1, 0): 1.0},
    ((1, 0), 1): {(1, 0): 1.0},
    ((0, 0), 8): {(0, 0): 1.0},
    ((0, 0), 4): {(1, 0): 1.0},
    ((0, 0), 2): {(0, 0): 1.0},
    ((0, 0), 1): {(0, 1): 1.0},
    ((0, 1), 8): {(0, 1): 1.0},
    ((0, 1), 4): {(0, 1): 1.0},
    ((0, 1), 2): {(0, 0): 1.0},
    ((0, 1), 1): {(0, 2): 1.0},
    ((0, 2), 8): {(0, 2): 1.0},
    ((0, 2), 4): {(1, 2): 1.0},
    ((0, 2), 2): {(0, 1): 1.0},
    ((0, 2), 1): {(0, 3): 1.0},
    ((2, 1), 8): {(2, 1): 1.0},
    ((2, 1), 4): {(2, 1): 1.0},
    ((2, 1), 2): {(2, 0): 1.0},
    ((2, 1), 1): {(2, 2): 1.0},
    ((2, 2), 8): {(1, 2): 1.0},
    ((2, 2), 4): {(2, 2): 1.0},
    ((2, 2), 2): {(2, 1): 1.0},
    ((2, 2), 1): {(2, 3): 1.0},
    ((2, 3), 8): {(1, 3): 1.0},
    ((2, 3), 4): {(2, 3): 1.0},
    ((2, 3), 2): {(2, 2): 1.0},
    ((2, 3), 1): {(2, 3): 1.0},
    ((1, 2), 8): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 4): {(2, 2): 1.0},
    ((1, 2), 2): {(1, 2): 1.0},
    ((1, 2), 1): {(1, 3): 1.0},
    }
    g.set(rewards, actions, probs)
    return g

def windy_grid_penalized(g, step_cost=-0.1):
    rewards = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0, 3): 1,
    (1, 3): -1
    }
    g.set(rewards, g.actions, g.probs)
    return g



if __name__ == '__main__':
    g = standard_grid()
    print(g.current_state())
    action = 4
    reward = g.move(action)
    print(g.current_state())
    print(g.game_over())
