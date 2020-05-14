import numpy as np
import random
from tdalg import TDAlg

class QLearn(TDAlg):
    def update(self):
        cum_reward = 0
        state = self.problem.starting_state
        while state not in self.problem.terminal_states:
            self.time += 1
            action = self.get_action(state)
            new_state, reward = self.problem.take_action(state, action)
            cum_reward += reward
            best = self.best_action(new_state)
            target = reward + self.gamma*self.qs[new_state][best] - self.qs[state][action]
            self.qs[state][action] += self.alpha*target
            state = new_state
        return cum_reward
    
    def update_from_model(self, s, a, s_, r):
        best = self.best_action(s_)
        target = r + self.gamma*self.qs[s_][best] - self.qs[s][a]
        self.qs[s][a] += self.alpha*target

