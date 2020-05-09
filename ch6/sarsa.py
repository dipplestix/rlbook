import numpy as np
import random

class Sarsa:
    def __init__(self, problem, qs=None, policy=None, eps=.05, gamma=1, alpha=1):
        self.problem = problem
        if qs is None:
            self.qs = {s: {a: 0 for a in problem.actions} for s in problem.states}
        else:
            self.qs = qs
        if policy is None:
            self.policy = {s: random.choice(problem.actions) for s in problem.states}
        else:
            self.policy = policy
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.state = None
    
    def get_action(self, state):
        if random.random() < self.eps:
            return random.choice(self.problem.actions)
        else:
            max_q = -np.inf
            for action in self.problem.actions:
                qval = self.qs[state][action]
                if  qval > max_q:
                    max_q = qval
                    best_action = action
            return best_action
            
            
    def run_episode(self):
        count = 0
        state = self.problem.starting_state
        action = self.get_action(state)
        while state not in self.problem.terminal_states:
            count += 1
            new_state, reward = self.problem.take_action(state, action)
            new_action = self.get_action(new_state)
            target = reward + self.gamma*self.qs[new_state][new_action] - self.qs[state][action]
            self.qs[state][action] += self.alpha*target
            action = new_action
            state = new_state
        return count
        
