import numpy as np
import random

class TDAlg_Model:
    def __init__(self, problem, n=10, qs=None, policy=None, eps=.05, gamma=1, alpha=1, max_time=None):
        self.problem = problem
        if qs is None:
            self.qs = {s: {a: 0 for a in problem.actions} for s in problem.states}
        else:
            self.qs = qs
        if policy is None:
            self.policy = {s: random.choice(problem.actions) for s in problem.states}
        else:
            self.policy = policy
        self.model = {}
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.n = n
        self.max_time = max_time
        self.state = None
        self.ep = 0
        self.time = 0
        self.ep_log = []
        self.r_log = []
    
    def get_action(self, state):
        if random.random() < self.eps:
            choice = random.choice(self.problem.actions)
        else:
            choice = self.best_action(state)
        return choice
        
    def best_action(self, state):
        choices = [a for a in self.problem.actions if self.qs[state][a] == max(self.qs[state].values())]
        best_action = random.choice(choices)
        return best_action
    
    def run_model(self):
        for i in range(self.n):
            s = random.choice(list(self.model.keys()))
            a = random.choice(list(self.model[s].keys()))
            s_, r = self.model[s][a]
            self.update_from_model(s, a, s_, r)
            
    def run_episode(self):
        if (self.max_time is not None and self.time <= self.max_time) or self.max_time is None:
            self.problem.update(self.ep)
            old_count = self.time
            cum_reward = self.update()
            self.ep_log.extend([self.ep for _ in range(self.time - old_count)])
            self.ep += 1
            if len(self.r_log) > 0:
                self.r_log.append(self.r_log[-1] + cum_reward)
            else:
                self.r_log.append(cum_reward)
            self.run_model()
#             for state in self.problem.states:
#                 self.policy[state] = self.best_action(state)
        else:
            pass
        