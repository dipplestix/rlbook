import numpy as np
import random
from tdalg_model import TDAlg_Model

class DynaQ(TDAlg_Model):
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
            try:
                self.model[state][action] = [new_state, reward]
            except:
                self.model[state] = {action: [new_state, reward]}
            state = new_state
        return cum_reward

    def update_from_model(self, s, a, s_, r):
        best = self.best_action(s_)
        target = r + self.gamma*self.qs[s_][best] - self.qs[s][a]
        self.qs[s][a] += self.alpha*target
