import numpy as np
from collections import Counter, defaultdict, deque, namedtuple

import gym
from gym import spaces

from config import config

import torch
import torch.nn.functional as F


class ObservationBuffer(gym.Env):
    '''
    used when not training. wrapper for Medical Env
    '''
    def __init__(self, target_obs_shape, buffer_len, warp_mode="bilinear"):
        '''
        Buffer observations and stack across channels (last axis).
        '''
        super(ObservationBuffer, self).__init__(self, target_obs_shape, len)
        self.target_width, self.target_height, self.target_depth = target_obs_shape
        self.target_size = torch.Size(1, 1, self.target_depth, self.target_height, self.target_width) # N x C x D x H x W
        self.buffer_len = buffer_len
        self.warp_mode = warp_mode

        self.obs_buffer = deque([], maxlen=buffer_len)
        self.obs_buffer_shape = self.obs_shape + (self.buffer_len,)
        self.observation_buffer_space = spaces.Box(low=0, high=255, shape=self.obs_buffer_shape, dtype=np.uint8)

    def reset(self, observation):
        '''
        Clear state buffer and re-initialized the last one with new observation.
        '''
        # set first k - 1 observations in buffer to be zeros
        for _ in range(self.k - 1):
            self.obs_buffer.append(np.zeros_like(observation))

        # add the current observation to the last one in the buffer
        self.obs_buffer.append(observation)

        return self.retrive()


    def warp(self, observation):
        '''
        Warp the observation into target obs_shape using 3D affine transformation
        '''
        # calculate the scale ratio along each of the 3 dimensions
        theta_x = 0
        theta_y = 0
        theta_z = 0

        # define the theta metric for affine transformation
        theta = torch.tensor([[theta_x, 0, 0, 0],
                              [0, theta_y, 0, 0],
                              [0, 0, theta_z, 0]], dtype=torch.float)

        # convert the observation into size [batch_size, channel, D, H, W]
        H, W, D = observation.size()
        observation = torch.permute(observation, (2, 0, 1)).view(1, 1, D, H, W)

        grid = F.affine_grid(theta=theta.unsqueeze(0), size=self.target_size)
        output = F.grid_sample(input=observation, grid=grid, mode=self.warp_mode)

        # convert the output shape back to H x W x D
        output = torch.permute(observation.view(self.target_depth, self.target_height, self.target_width), (1, 2, 0))

        return output

    def append(self, observation):
        '''
        Append a new observation to the buffer after warp it to target obs_shape
        '''
        warped_observation = self.warp(observation)
        self.obs_buffer.append(warped_observation)

    def retrive(self):
        '''
        Return the stacked observations from the buffer
        '''
        assert len(self.obs_buffer) == self.buffer_len
        return torch.stack(self.obs_buffer, axis=-1)


class Medical_Env(gym.Env):
    '''
    Class for the 3D CT Scan Environment
    A each step the agent will take an action, and the environment 
    will return the updated observation and rewards.
    '''
    def __init__(self, dataloader, player, mode, obs_dims, history_length, max_steps, alpha=0.1, tau=0.85, state_offset=16):
        super(Medical_Env, self).__init__(dataloader, player, mode, history_length, max_steps, alpha, tau, state_offset)

        self.dataloader = dataloader
        self.player = player
        self.mode = mode

        self.max_steps = max_steps
        self.obs_dims = obs_dims
        self.dims = len(self.obs_dims)
        self.width, self.height, self.depth = self.obs_dims
        self.hisotry_length = history_length
        self.alpha = alpha # how large the action should be for each step
        self.tau = tau # IoU threshold
        self.state_offset = state_offset # how much additional space to include in the state
        self.inital_box_coverage = config.inital_box_coverage # specify the percentage of image volumne the inital bounding box should cover

        self.box_history = [(0,) * self.dims] * self.history_length

        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_dims,dtype=np.uint8)

        self.observation_buffer = ObservationBuffer(obs_shape=self.obs_dims, len=self.hisotry_length)

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
        self.observation_buffer.reset()

    def start_new_episode(self):
        '''
        start a new episode and load an environment from the dataloader
        the initial state of the bounding box is defined as the entire scan
        (bx; by; bz): the top-left-front corner 
        (bw; bh; bd): the lower-right-back
        '''
        # reset the environment
        self.reset()

        # load a new scan from the dataloader
        self.image, self.label = self.dataloader.sample_scan_circular()
        self.image_width, self.image_height, self.image_depth = self.image.size()
        self.label_box = ((self.label[0], self.label[1], self.label[2]), (self.label[3], self.label[4], self.label[5]))

        # randomly initialize the bounding box to include a large percentage of the scan's volumne centered in the middle of the scan
        self.current_box = self.initalize_bounding_box()
        self.current_iou = self.calculaue_IoU(self.current_box, self.label_box)

    def initalize_bounding_box(self):
        '''
        initialize the bounding box which covers a large percentage of the scan centered in the middle
        the exact percentage is specificed by self.inital_box_coverage
        if it is a range, then randomly choose a number from this range as the coverage percentage
        '''
        # if self.inital_box_coverage is a fixed percentage
        if len(self.inital_box_coverage) == 1:
            cover_percent = self.inital_box_coverage
        # if self.inital_box_coverage is a range
        elif len(self.inital_box_coverage) == 2:
            [low, high] = self.inital_box_coverage * 100
            cover_percent = np.random.randint(low, high) / 100
        
        # calculate the bounding box coordinate that include the cover_percent of original scan
        x = self.image_width * (1 - cover_percent) / 2
        y = self.image_height * (1 - cover_percent) / 2
        z = self.image_depth * (1 - cover_percent) / 2

        w = self.image_width - x
        h = self.image_height - y
        d = self.image_depth - z 

        return ((x, y, z), (w, h, d))


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
            x = x - dw / 2
            w = w + dw / 2
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
        self.observation_buffer.append(self.current_state)

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

        # retrive the current + previous observation state from the buffer
        obs_states = self.observation_buffer.retrive()

        return obs_states, self.reward, self.terminal, info

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
        crop the image based on the new bounding box and include additional context surrounding the box
        based on the self.state_offset
        '''
        ((x, y, z), (w, h, d)) = self.current_box

        # calculate the coordinate of state in the image after dilating the bounding box 
        # by self.state_offset pixels in all 3 dimension
        sx = torch.min(0, x - self.state_offset)
        sy = torch.min(0, y - self.state_offset)
        sz = torch.min(0, z - self.state_offset)

        sw = torch.max(self.image_width, w + self.state_offset)
        sh = torch.max(self.image_height, h + self.state_offset)
        sd = torch.max(self.image_depth, d + self.state_offset)

        # extract the state from the image
        current_state = self.image[sx:sw, sy:sh, sz:sd]

        return current_state

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