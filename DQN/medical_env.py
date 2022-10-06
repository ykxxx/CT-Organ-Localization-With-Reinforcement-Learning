import numpy as np
from collections import Counter, defaultdict, deque, namedtuple

import gym
from gym import spaces

from config import config


class Medical_Env(gym.Env):
    '''
    Class for the 3D CT Scan Environment
    A each step the agent will take an action, and the environment 
    will return the updated observation and rewards.
    '''
    def __init__(self, dataloader, player, mode, env_dims, history_length, max_steps, alpha, tau):
        super(Medical_Env, self).__init__(dataloader, player, mode, history_length, max_steps, alpha, tau)

        self.dataloader = dataloader
        self.player = player
        self.mode = mode

        self.max_steps = max_steps
        self.env_dims = env_dims
        self.dims = len(self.env_dims)
        self.width, self.height, self.depth = self.env_dims
        self.hisotry_length = history_length
        self.alpha = alpha
        self.tau = tau

        self.box_history = [(0,) * self.dims] * self.history_length

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.screen_dims,
                                            dtype=np.uint8)

    def reset(self):
        '''
        reset current eposide
        '''
        self.terminal = False
        self.reward = 0
        self.step_count = 0
        self.box_history = []
        self.iou_history = []
        self.reward_history = []

    def start(self):
        '''
        start a new episode and load an environment from the dataloader
        the initial state of the bounding box is defined as the entire scan
        (bx; by; bz): the top-left-front corner 
        (bw; bh; bd): the lower-right-back
        '''
        self.image, self.label = self.dataloader.sample_scan_circular()
        self.label_box = ((self.label[0], self.label[1], self.label[2]), (self.label[3], self.label[4], self.label[5]))

        top_left_front = (0, 0, 0)
        lower_right_back = self.env_dims

        self.current_box = (top_left_front, lower_right_back)
        self.current_iou = self.calculaue_IoU(self.current_box, self.label_box)

    def step(self, action):
        '''
        take the selected action in the current environment
        update environment after the action and return a reward
        11 total possible action: 
            - translation: tx+, tx-, ty+, ty-, tz+, tz-
            - scaling: s+, s-
            - deformation: dx (thinner), dy (flatter), dz (taller)
        '''
        current_box = self.current_box
        current_iou = self.current_iou

        ((x, y, z), (w, h, d)) = current_box

        # translation tx+
        if action == 0:
            tw = self.alpha * (x - w)
            x = x + tw
            w = w + tw
        # translation tx-
        elif action == 1:
            tw = self.alpha * (x - w)
            x = x - tw
            w = w - tw
        # translation ty+
        elif action == 2:
            th = self.alpha * (y - h)
            y = y + th
            h = h + th
        # translation ty-
        elif action == 3:
            th = self.alpha * (y - h)
            y = y - th
            h = h - th
        # translation tz+
        elif action == 4:
            td = self.alpha * (z - d)
            z = z + td
            d = d + td
        # translation tz-
        elif action == 5:
            td = self.alpha * (z - d)
            z = z - td
            d = d - td
        # scaling s+
        elif action == 6:
            sw = self.alpha * abs(x - w)
            x = x + sw / 2
            w = w - sw / 2
            sh = self.alpha * abs(y - h)
            y = y + sh / 2
            h = h - sh / 2
            sd = self.alpha * abs(z - d)
            z = z + sd / 2
            d = d - sd / 2
        # scaling s-
        elif action == 7:
            sw = self.alpha * abs(x - w)
            x = x - sw / 2
            w = w + sw / 2
            sh = self.alpha * abs(y - h)
            y = y - sh / 2
            h = h + sh / 2
            sd = self.alpha * abs(z - d)
            z = z - sd / 2
            d = d + sd / 2
        # deformation dx (thinner)
        elif action == 8:
            dw = self.alpha * abs(x - w)
            x = x - sw / 2
            w = w + sw / 2
        # deformation dy (flatter)
        elif action == 9:
            dh = self.alpha * abs(y - h)
            y = y - dh / 2
            h = h + dh / 2
        # deformation dz (taller)
        elif action == 10:
            dd = self.alpha * abs(z - d)
            z = z - dd / 2
            d = d + dd / 2
        
        new_box = ((x, y, z), (w, h, d))
        new_iou = self.calculaue_IoU(current_box, new_box)

        # calculate reward for the action
        # reward = +1 if the action increase the IoU
        self.reward = self._calculate_reward(current_iou, new_iou)

        # update environment variables
        self.current_box = new_box
        self.current_iou = new_iou
        self.current_state = self._get_current_state()

        # update environment history
        self._update_history()

        # check if terminal condition is met
        # during training, terminate if IoU > tau
        if self.mode == "train" and self.current_iou > self.tau:
            self.terminal = True
        # during testing, terminate if oscillation occurs
        elif self.mode == "test" and self._oscillate():
            self.terminal = True

        info = {'terminal': self.terminal,
                        'IoU': self.current_iou,
                        'reward': self.reward}

        return self._get_current_state(), self.reward, self.terminal, info

    def calculaue_IoU(self, box1, box2):
        '''
        calculate the IoU of two 3D bounding boxes
        box1: ((x1, y1, z1), (w1, h1, d1))
        box2: ((x2, y2, z2), (w2, h2, d2))
        '''

        ((x1, y1, z1), (w1, h1, d1)) = box1
        ((x2, y2, z2), (w2, h2, d2)) = box2

        # calculate overlap volumn
        intersect_v = max(min(x1, x2) - max(w1, w2), 0) * max(min(y1, y2) - max(h1, h2), 0) * max(min(z1, z2) - max(d1, d2), 0)

        # calculate intersection volumn
        box1_v = abs(x1 - w1) * abs(y1 - h1) * abs(z1 - d1)
        box2_v = abs(x2 - w2) * abs(y2 - h2) * abs(z2 - d2)
        union_v = box1_v + box2_v - intersect_v

        return intersect_v / union_v

    def _get_current_state(self):
        '''
        crop the image based on the new bounding box to update the state
        '''
        ## TODO: how is the state extracted from the image?
        return

    def _update_history(self):
        '''
        update the history buffer with the current state
        '''
        self.box_history.append(self.current_box)
        self.iou_history.append(self.current_iou)
        self.reward_history.append(self.reward)

    def _calculate_reward(self, current_iou, new_iou):
        '''
        calculate the reward for action based on current_iou and new_iou after the action
        '''
        return 1 if new_iou > current_iou else -1

    def _oscillate(self):
        '''
        check if the agent is stuck and oscillating
        adopted method proposed in @Alansary et al., 2019
        '''
        counter = Counter(self.box_history)
        freq = counter.most_common()

        if freq[0][0] == (0, 0, 0):
            if (freq[1][1] > 3):
                return True
            else:
                return False
        elif (freq[0][1] > 3):
            return True
