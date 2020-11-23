from collections import deque
import numpy as np


class MarketEnv:
    def __init__(self, sales_model, prime_cost):
        self.sales_model = sales_model # Linear Regression
        self.prime_cost = prime_cost # 100 for every instance
    
    def reset(self):
        self.t = 1
        return self.t
    
    def step(self, action):
        sales = self.sales_model.predict(self.t, action)
        profit = sales * (action - self.prime_cost)
        self.t += 1

        return self.t, profit, self.t == 53


# Since tabular Q-learning deals with finite state-space, the agent should discretize continious observations.
class MarketAgent:
    def __init__(self, min_price, max_price, bins_number, learning_rate=0.1, discount_factor=0.96,
                 exploration_rate=0.98, exploration_decay_rate=0.99):
        # TODO: Your code here
        self.min_price = min_price
        self.max_price = max_price
        self.bins_number = bins_number
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None # week number
        self.action = None # price
        self.memory = np.zeros((52, self.bins_number))
    
    def begin_episode(self, observation):
        # TODO: Your code here. Implement method that returns an epsilon-greedy action.
        # return action - price
        # observation is state - num of week
        self.exploration_rate *= self.exploration_decay_rate # change epsilon due to the distribution
        
        self.state = observation % 52
        bin_num = np.argmax(self.memory[self.state])
        self.action = bin_num
        
        return int(self.action / self.bins_number * (self.max_price - self.min_price) + self.min_price) 
    
    def act(self, observation, reward, done):
        # TODO: Your code here. Implement method that returns an epsilon-greedy action and updates Q-table.
        next_state = observation % 52
        if np.random.uniform(0, 1) <= self.exploration_rate:
            next_action = np.random.randint(self.bins_number)
        else:
            next_action = np.argmax(self.memory[next_state])
        
        self.memory[self.state, self.action] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.memory[next_state]) - self.memory[self.state, self.action])

        self.state = next_state
        self.action = next_action
        
        return int(self.action / self.bins_number * (self.max_price - self.min_price) + self.min_price) 
