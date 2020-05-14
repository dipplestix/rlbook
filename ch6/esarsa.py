import numpy as np
import random
from tdalg import TDAlg

class ESarsa(TDAlg):
    def update(self):
        cum_reward = 0
        state = self.problem.starting_state
        while state not in self.problem.terminal_states:
            self.time += 1
            action = self.get_action(state)
            new_state, reward = self.problem.take_action(state, action)
            val = reward - self.qs[state][action]
            best = self.best_action(new_state)
            for a in self.problem.actions:
                if a == best:
                    val += (1-self.eps)*self.gamma*self.qs[new_state][a]
                val += (self.eps/len(self.problem.actions))*self.gamma*self.qs[new_state][a]
            target = val
            cum_reward += reward
            self.qs[state][action] += self.alpha*target
            state = new_state        
        return cum_reward
