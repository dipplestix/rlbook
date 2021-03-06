import numpy as np
import random
from tdalg import TDAlg

class Sarsa(TDAlg):
    def update(self):
        cum_reward = 0
        state = self.problem.starting_state
        action = self.get_action(state)
        while state not in self.problem.terminal_states:
            self.time += 1
            new_state, reward = self.problem.take_action(state, action)
            new_action = self.get_action(new_state)
            target = reward + self.gamma*self.qs[new_state][new_action] - self.qs[state][action]
            cum_reward += reward
            self.qs[state][action] += self.alpha*target
            action = new_action
            state = new_state        
        return cum_reward
