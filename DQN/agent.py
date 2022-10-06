import torch
import random
import numpy as np

from config import config

class Agent:
    def __init__(self, mode, action_size):
        self.mode = mode
        self.action_size = action_size
    
    def choose_action(self, q_values, epsilon):

        # during training, use epsilon-greedy action selection
        if self.mode == "train":
            if random.random() > epsilon:
                action = np.argmax(q_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
        # during testing, select the action with the highest q-value
        else:
            action = np.argmax(q_values.cpu().data.numpy())

        return action

        