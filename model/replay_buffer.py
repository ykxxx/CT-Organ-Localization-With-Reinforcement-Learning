from collections import namedtuple, deque

from config import config

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Replay_Buffer(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)