import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque, namedtuple
import copy

import gym
from gym import spaces

from config import config

import torch
import torch.nn.functional as F

from dataloader import CT_DataLoader

Rectangle = namedtuple('Rectangle', ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])


class MedicalPlayer(gym.Env):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(self, train_mode=False,
                 screen_dims=(45, 45, 45), history_length= 4, action_step=10,
                 max_num_frames=1000, agents=1, reward_method='binary',
                 oscillations_allowed=4, logger=None):
        """
        :param train_directory: environment or game name
        :param viz: visualization
            set to 0 to disable
            set to +ve number to be the delay between frames to show
            set to a string to be the directory for storing frames
        :param screen_dims: shape of the frame cropped from the image to feed
            it to dqn (d,w,h) - defaults (27,27,27)
        :param nullop_start: start with random number of null ops
        :param location_history_length: consider lost of lives as end of
            episode (useful for training)
        :max_num_frames: maximum numbe0r of frames per episode.
        """
        super(MedicalPlayer, self).__init__()
        self.agents = agents
        self.oscillations_allowed = oscillations_allowed
        self.logger = logger
        # inits stat counters+
        self.reset_stat()

        # counter to limit number of steps per episodes
        self.cnt = 0
        # maximum number of frames (steps) per episodes
        self.max_num_frames = max_num_frames
        # stores information: terminal, score, distError
        self.info = None
      
        # training flag
        self.train_mode = train_mode

        # image dimension (3D: (45, 45, 45))
        self.screen_dims = screen_dims 
        self.dims = len(self.screen_dims)
        
        # multi-scale agent
        self.action_step = action_step

        # init env dimensions
        self.width, self.height, self.depth = screen_dims

        # how to calculate reward
        self.reward_method = reward_method

        # stat counter to store current score or accumlated reward
        # self.current_episode_score = [StatCounter()] * self.agents

        # get action space and minimal action set
        self.action_space = spaces.Discrete(6)  # change number actions here
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.screen_dims,
                                            dtype=np.uint8)

        # history buffer for storing last locations to check oscilations
        self._history_length = history_length
        self._loc_history = [
            [((0,) * self.dims, (0,) * self.dims) for _ in range(self._history_length)]
            for _ in range(self.agents)]
        self._qvalues_history = [
            [(0,) * self.actions for _ in range(self._history_length)]
            for _ in range(self.agents)]
            
        # initialize rectangle limits from input image coordinates
        # self.rectangle = [Rectangle(0, 0, 0, 0, 0, 0)] * int(self.agents)

        # add data loader
        self.files = CT_DataLoader()

        # prepare file sampler
        self.filepath = None
        
        # reset buffer, terminal, counters, and init new_random_game
        self._restart_episode()

    def reset(self):
        # with _ALE_LOCK:
        self._restart_episode()
        return self._current_state()

    def _restart_episode(self):
        """
        restart current episode
        """
        self.sampled_files = self.files.sample_circular()
        self.terminal = [False] * self.agents
        self.reward = np.zeros((self.agents,))
        self.cnt = 0  # counter to limit number of steps per episodes
        # self.num_games.feed(1)

        self._loc_history = [
            [((0,) * self.dims, (0,) * self.dims) for _ in range(self._history_length)]
            for _ in range(self.agents)]

        # list of q-value lists
        self._qvalues_history = [
            [(0,) * self.actions for _ in range(self._history_length)]
            for _ in range(self.agents)]

        # for i in range(0, self.agents):
        #     self.current_episode_score[i].reset()

        self.new_random_game()

    def new_random_game(self):
        """
        load image,
        set dimensions,
        randomize start point,
        init _screen, qvals,
        calc distance to goal
        """
        self.terminal = [False] * self.agents
        self.viewer = None

        # sample a new image
        self._image, self._target_loc  = self.sampled_files

        # print(self._image.min(), self._image.max())
        # TODO: add it as a parameter to the class
        cover_percent = 0.5

        # H * W* D
        batch_size, self.image_height, self.image_width, self.image_depth = self._image.shape
        
        # start coordinates pf bounding box
        x1 = [int(self.image_height * round((1 - cover_percent), 1)/2)] * self.agents
        y1 = [int(self.image_width * round((1 - cover_percent), 1)/2)] * self.agents
        z1 = [int(self.image_depth * round((1 - cover_percent), 1)/2)] * self.agents

        # end coordiates of bounding box
        x2 = [self.image_height - val for val in x1]
        y2 = [self.image_width - val for val in y1]
        z2 = [self.image_depth - val for val in z1]
        
        # image volume size
        self._image_dims = self._image[0].shape

        #######################################################################

        #######################################################################

        self._location = [((x1[i], y1[i], z1[i]), (x2[i], y2[i], z2[i])) for i in range(self.agents)]
        self._box_image = [self._image[:, self._location[i][0][0]:self._location[i][1][0]+1, self._location[i][0][1]:self._location[i][1][1]+1, self._location[i][0][2]:self._location[i][1][2]+1].squeeze(0) for i in range(self.agents)]
        self._start_location = [((x1[i], y1[i], z1[i]), (x2[i], y2[i], z2[i])) for i in range(self.agents)]
        self._qvalues = [[0, ] * self.actions] * self.agents
        
        self._screen = self._current_state()

        if self.train_mode:
            self.cur_dist = [
                self.calcIou(
                    # location varies for agents, but target loc is fixed
                    self._location[i],
                    self._target_loc) 
                    for i in range(self.agents)]
        else:
            self.cur_dist = [0, ] * self.agents

    def calcIou(self, box1, box2):
        """ 
        calculate the IOU between two bounding boxes in mm
        box1: current loc of bounding box
        box2: target loc of bounding box
        """
      
        (start_c, end_c) = box1 
        (x1_c, y1_c, z1_c) = start_c
        (x2_c, y2_c, z2_c) = end_c
    
        (start_t, end_t) = box2
        (x1_t, y1_t, z1_t) = start_t
        (x2_t, y2_t, z2_t) = end_t
        
        x_diff = max(min(x2_t, x2_c) - max(x1_t, x1_c), 0)
        y_diff = max(min(y2_t, y2_c) - max(y1_t, y1_c), 0)
        z_diff = max(min(z2_t, z2_c) - max(z1_t, z1_c), 0)

        intersect_v = x_diff * y_diff * z_diff
        box1_v = abs(x2_c - x1_c) * abs(y2_c - y1_c) * abs(z2_c - z1_c)
        box2_v = abs(x2_t - x1_t) * abs(y2_t - y1_t) * abs(z2_t - z1_t)

        union_v = box1_v + box2_v - intersect_v

        # print(intersect_v/union_v)
        # print("Iou Debug")

        return intersect_v/union_v


    def step(self, act, q_values, isOver):
        """The environment's step function returns exactly what we need.
        Args:
          act:
        Returns:
          observation (object):
            an environment-specific object representing your observation of
            the environment. For example, pixel data from a camera, joint
            angles and joint velocities of a robot, or the board state in a
            board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies
            between environments, but the goal is always to increase your total
            reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be
            useful for learning (for example, it might contain the raw
            probabilities behind the environment's last state change). However,
            official evaluations of your agent are not allowed to use this for
            learning.
        """
        self._qvalues = q_values
        current_loc = self._location
        next_location = copy.deepcopy(current_loc)

        self.terminal = [False] * self.agents
        go_out = [False] * self.agents

        # agent i movement
        for i in range(self.agents):
            # UP Z+ -----------------------------------------------------------
            if (act[i] == 0):

                next_location[i] = ((
                    current_loc[i][0][0],current_loc[i][0][1],
                    round(current_loc[i][0][2] +self.action_step)),
                    (current_loc[i][1][0], current_loc[i][1][1],
                    round(current_loc[i][1][2] +self.action_step) ))
                
                if (next_location[i][1][2] >= self._image_dims[2]):
                    # print(' trying to go out the image Z+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True

            # FORWARD Y+ ------------------------------------------------------
            if (act[i] == 1):
                next_location[i] = ((
                    current_loc[i][0][0],
                    round(current_loc[i][0][1] + self.action_step),
                    current_loc[i][0][2]),
                    (
                    current_loc[i][1][0],
                    round(current_loc[i][1][1] + self.action_step),
                    current_loc[i][1][2]))
                if (next_location[i][1][1] >= self._image_dims[1]):
                    # print(' trying to go out the image Y+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # RIGHT X+ --------------------------------------------------------
            if (act[i] == 2):
                next_location[i] = ((
                    round(current_loc[i][0][0] +self.action_step),
                    current_loc[i][0][1],
                    current_loc[i][0][2]),
                    (round(current_loc[i][1][0] +self.action_step),
                    current_loc[i][1][1],
                    current_loc[i][1][2]) )
                if next_location[i][1][0] >= self._image_dims[0]:
                    # print(' trying to go out the image X+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # LEFT X- ---------------------------------------------------------
            if act[i] == 3:
                next_location[i] = ((
                    round(current_loc[i][0][0] - self.action_step),
                    current_loc[i][0][1],
                    current_loc[i][0][2]),
                    (round(current_loc[i][1][0]  - self.action_step),
                    current_loc[i][1][1],
                    current_loc[i][1][2]) )
                if next_location[i][0][0] <= 0:
                    # print(' trying to go out the image X- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # BACKWARD Y- -----------------------------------------------------
            if act[i] == 4:
                next_location[i] = ((
                    current_loc[i][0][0],
                    round(current_loc[i][0][1] -self.action_step),
                    current_loc[i][0][2]),
                    (
                    current_loc[i][1][0],
                    round(current_loc[i][1][1] -self.action_step),
                    current_loc[i][1][2]))
                
                if next_location[i][0][1] <= 0:
                    # print(' trying to go out the image Y- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # DOWN Z- ---------------------------------------------------------
            if act[i] == 5:
                next_location[i] = ((
                    current_loc[i][0][0], current_loc[i][0][1],
                    round(current_loc[i][0][2] - self.action_step)),
                    (current_loc[i][1][0], current_loc[i][1][1],
                    round(current_loc[i][1][2] - self.action_step) ))
                if next_location[i][0][2] <= 0:
                    # print(' trying to go out the image Z- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # scaling s+ ---------------------------------------------------------
            if act[i] == 6:
                next_location[i] = ((
                    round(current_loc[i][0][0] - self.action_step/2),
                    round(current_loc[i][0][1] - self.action_step/2),
                    round(current_loc[i][0][2] - self.action_step/2)),
                    (round(current_loc[i][1][0] + self.action_step/2),
                     round(current_loc[i][1][1] + self.action_step/2),
                     round(current_loc[i][1][2] + self.action_step/2),
                     ))
                if (next_location[i][1][2] >= self._image_dims[2] or 
                    next_location[i][1][1] >= self._image_dims[1] or 
                    next_location[i][1][0] >= self._image_dims[0] or
                    next_location[i][0][0] <= 0 or 
                    next_location[i][0][1] <= 0 or 
                    next_location[i][0][2] <= 0):
                    
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # scaling s- ---------------------------------------------------------
            if act[i] == 7:
                next_location[i] = ((
                    round(current_loc[i][0][0] + self.action_step/2),
                    round(current_loc[i][0][1] + self.action_step/2),
                    round(current_loc[i][0][2] + self.action_step/2)),
                    (round(current_loc[i][1][0] - self.action_step/2),
                     round(current_loc[i][1][1] - self.action_step/2),
                     round(current_loc[i][1][2] - self.action_step/2),
                     ))
                if (next_location[i][1][2] >= self._image_dims[2] or 
                    next_location[i][1][1] >= self._image_dims[1] or 
                    next_location[i][1][0] >= self._image_dims[0] or
                    next_location[i][0][0] <= 0 or 
                    next_location[i][0][1] <= 0 or 
                    next_location[i][0][2] <= 0):
                   
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # deformation dx (thinner) ---------------------------------------------------------
            if act[i] == 8:
                next_location[i] = ((
                    round(current_loc[i][0][0] - self.action_step/2),
                    round(current_loc[i][0][1]),
                    round(current_loc[i][0][2])),
                    (round(current_loc[i][1][0] + self.action_step/2),
                     round(current_loc[i][1][1]),
                     round(current_loc[i][1][2]),
                     ))
                if (next_location[i][1][0] >= self._image_dims[0] or 
                    next_location[i][0][0] <= 0 
                    ):
                    # print(' trying to go out the image x- or x+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # deformation dy (flatter)---------------------------------------------------------
            if act[i] == 9:
                next_location[i] = ((
                    round(current_loc[i][0][0]),
                    round(current_loc[i][0][1] - self.action_step/2),
                    round(current_loc[i][0][2])),
                    (round(current_loc[i][1][0]),
                     round(current_loc[i][1][1] + self.action_step/2),
                     round(current_loc[i][1][2]),
                     ))
                if (next_location[i][1][1] >= self._image_dims[1] or 
                    next_location[i][0][1] <= 0 
                    ):
                    # print(' trying to go out the image y- or y+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            # deformation dz (taller) ---------------------------------------------------------
            if act[i] == 10:
                next_location[i] = ((
                    round(current_loc[i][0][0] ),
                    round(current_loc[i][0][1] ),
                    round(current_loc[i][0][2] - self.action_step/2)),
                    (round(current_loc[i][1][0] ),
                     round(current_loc[i][1][1] ),
                     round(current_loc[i][1][2] + self.action_step/2),
                     ))
                if (next_location[i][1][2] >= self._image_dims[2] or 
                    next_location[i][0][2] <= 0 
                    ):
                    # print(' trying to go out the image Z- or z+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
            
            # -----------------------------------------------------------------

        #######################################################################

        if self.train_mode:
            for i in range(self.agents):
                # punish -1 reward if the agent tries to go out
                if go_out[i]:
                    self.reward[i] = -1
                else:
                    self.reward[i] = self._calc_reward(current_loc[i], next_location[i])

        # update screen, curr_image, reward ,location, terminal
        self._location = next_location
        self._box_image = [self._image[:, self._location[i][0][0]:self._location[i][1][0]+1, self._location[i][0][1]:self._location[i][1][1]+1, self._location[i][0][2]:self._location[i][1][2]+1].squeeze(0) for i in range(self.agents)]
        self._screen = self._current_state()

        # terminate if the distance is less than 1 during trainig
        if self.train_mode:
            for i in range(self.agents):
                if self.cur_dist[i] > 0.8:
                    print(f"IOU of agent {i} is >= 0.8")
                    self.terminal[i] = True
                    # self.num_success[i].feed(1)

        """
        # terminate if maximum number of steps is reached
        self.cnt += 1
        if self.cnt >= self.max_num_frames:
            for i in range(self.agents):
                self.terminal[i] = True
        """

        # update history buffer with new location and qvalues
        if self.train_mode:
            for i in range(self.agents):
                self.cur_dist[i] = self.calcIou(self._location[i], self._target_loc)

        self._update_history()
        # check if agent oscillates
        # if self._oscillate:
        #     self._location = self.getBestLocation()
        #     # self._location=[item for sublist in temp for item in sublist]
        #     self._screen = self._current_state()

        #     if self.task != 'play':
        #         for i in range(self.agents):
        #             self.cur_dist[i] = self.calcIou(self._location[i],
        #                                                  self._target_loc)

        #     # multi-scale steps
        #     if self.multiscale:
        #         if self.xscale > 1:
        #             self.xscale -= 1
        #             self.yscale -= 1
        #             self.zscale -= 1
        #             self.action_step = int(self.action_step / 3)
        #             self._clear_history()
        #         # terminate if scale is less than 1
        #         else:
        #             for i in range(self.agents):
        #                 self.terminal[i] = True
        #                 # if self.cur_dist[i] <= 1:
        #                 #     self.num_success[i].feed(1)
        #     else:
        #         for i in range(self.agents):
        #             self.terminal[i] = True
        #             # if self.cur_dist[i] <= 1:
        #             #     self.num_success[i].feed(1)
       

        distance_error = self.cur_dist
        # for i in range(self.agents):
        #     self.current_episode_score[i].feed(self.reward[i])

        info = {}
        for i in range(self.agents):
            # info[f"score_{i}"] = self.current_episode_score[i].sum
            info[f"gameOver_{i}"] = self.terminal[i]
            info[f"distError_{i}"] = distance_error[i]
           
            info[f"agent_xpos_{i}"] = (self._location[i][0][0], self._location[i][1][0]) 
            info[f"agent_ypos_{i}"] = (self._location[i][0][1], self._location[i][1][1]) 
            info[f"agent_zpos_{i}"] = (self._location[i][0][2], self._location[i][1][2]) 
            info[f"landmark_xpos_{i}"] = (self._target_loc[0][0], self._target_loc[1][0])
            info[f"landmark_ypos_{i}"] = (self._target_loc[0][1], self._target_loc[1][1])
            info[f"landmark_zpos_{i}"] = (self._target_loc[0][2], self._target_loc[1][2])
        return self._screen, self.reward, self.terminal, info

    def getBestLocation(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_locations = []
        for i in range(self.agents):
            last_qvalues_history = self._qvalues_history[i][-4:]
            last_loc_history = self._loc_history[i][-4:]
            best_qvalues = np.max(last_qvalues_history, axis=1)
            best_idx = best_qvalues.argmin()
            best_locations.append(last_loc_history[best_idx])
        return best_locations

    def _clear_history(self):
        ''' clear history buffer with current states
        '''
        self._loc_history = [
            [((0,) * self.dims, (0,) * self.dims) for _ in range(self._history_length)]
            for _ in range(self.agents)]
        self._qvalues_history = [
            [(0,) * self.actions for _ in range(self._history_length)]
            for _ in range(self.agents)]

    def _update_history(self):
        ''' update history buffer with current states
        '''
        for i in range(self.agents):
            # update location history
            self._loc_history[i].pop(0)
            self._loc_history[i].insert(
                len(self._loc_history[i]), self._location[i])

            # update q-value history
            self._qvalues_history[i].pop(0)
            self._qvalues_history[i].insert(
                len(self._qvalues_history[i]), self._qvalues[i])

    def _current_state(self):
        """
        wrap image data around current location to update what network sees.
        update rectangle
        :return: new state
        """
        # initialize screen with zeros - all background
        screen = np.zeros(
            (self.agents,
             self.screen_dims[0],
             self.screen_dims[1],
             self.screen_dims[2]))

        theta_x = 1
        theta_y = 1
        theta_z = 1

        # define the theta metric for affine transformation
        theta = torch.tensor([[theta_x, 0, 0, 0],
                              [0, theta_y, 0, 0],
                              [0, 0, theta_z, 0]], dtype=torch.float)

        # convert the observation into size [batch_size, channel, D, H, W]
        for i in range(self.agents):
          H = self._location[i][1][0] - self._location[i][0][0] + 1# x2[i] - x1[i]
          W = self._location[i][1][1] - self._location[i][0][1] + 1# y2[i] - y1[i]
          D = self._location[i][1][2] - self._location[i][0][2] + 1# z2[i] - z1[i]

          # print(self._box_image[i].min(), self._box_image[i].max() )
          # print(self._box_image)
          observation = torch.permute(self._box_image[i], (2, 0, 1)).view(1, 1, D, H, W)
          grid = F.affine_grid(theta=theta.unsqueeze(0), size=(1, 1) + self.screen_dims)
          output = F.grid_sample(input=observation, grid=grid, mode= "bilinear")
          # print(output)
          # convert the output shape back to H x W x D
          screen[i] = torch.permute(output.view(self.screen_dims[0], self.screen_dims[1], self.screen_dims[2]), (1, 2, 0))
          # self.rectangle[i] = Rectangle(xmin, xmax,
          #                                 ymin, ymax,
          #                                 zmin, zmax)
        return screen
        

        # for i in range(self.agents):
        #     # screen uses coordinate system relative to origin (0, 0, 0)
        #     screen_xmin, screen_ymin, screen_zmin = 0, 0, 0
        #     screen_xmax, screen_ymax, screen_zmax = self.screen_dims

        #     # extract boundary locations using coordinate system relative to
        #     # "global" image
        #     # width, height, depth in terms of screen coord system

        #     if self.xscale % 2:
        #         xmin = self._location[i][0] - \
        #             int(self.width * self.xscale / 2) - 1
        #         xmax = self._location[i][0] + int(self.width * self.xscale / 2)
        #         ymin = self._location[i][1] - \
        #             int(self.height * self.yscale / 2) - 1
        #         ymax = self._location[i][1] + \
        #             int(self.height * self.yscale / 2)
        #         zmin = self._location[i][2] - \
        #             int(self.depth * self.zscale / 2) - 1
        #         zmax = self._location[i][2] + int(self.depth * self.zscale / 2)
        #     else:
        #         xmin = self._location[i][0] - \
        #             round(self.width * self.xscale / 2)
        #         xmax = self._location[i][0] + \
        #             round(self.width * self.xscale / 2)
        #         ymin = self._location[i][1] - \
        #             round(self.height * self.yscale / 2)
        #         ymax = self._location[i][1] + \
        #             round(self.height * self.yscale / 2)
        #         zmin = self._location[i][2] - \
        #             round(self.depth * self.zscale / 2)
        #         zmax = self._location[i][2] + \
        #             round(self.depth * self.zscale / 2)

        #     ###########################################################

        #     # check if they violate image boundary and fix it
        #     if xmin < 0:
        #         xmin = 0
        #         screen_xmin = screen_xmax - \
        #             len(np.arange(xmin, xmax, self.xscale))
        #     if ymin < 0:
        #         ymin = 0
        #         screen_ymin = screen_ymax - \
        #             len(np.arange(ymin, ymax, self.yscale))
        #     if zmin < 0:
        #         zmin = 0
        #         screen_zmin = screen_zmax - \
        #             len(np.arange(zmin, zmax, self.zscale))
        #     if xmax > self._image_dims[0]:
        #         xmax = self._image_dims[0]
        #         screen_xmax = screen_xmin + \
        #             len(np.arange(xmin, xmax, self.xscale))
        #     if ymax > self._image_dims[1]:
        #         ymax = self._image_dims[1]
        #         screen_ymax = screen_ymin + \
        #             len(np.arange(ymin, ymax, self.yscale))
        #     if zmax > self._image_dims[2]:
        #         zmax = self._image_dims[2]
        #         screen_zmax = screen_zmin + \
        #             len(np.arange(zmin, zmax, self.zscale))

        #     # crop image data to update what network sees
        #     # image coordinate system becomes screen coordinates
        #     # scale can be thought of as a stride
        #     screen[i,
        #            screen_xmin:screen_xmax,
        #            screen_ymin:screen_ymax,
        #            screen_zmin:screen_zmax] = self._image[i].data[
        #         xmin:xmax:self.xscale,
        #         ymin:ymax:self.yscale,
        #         zmin:zmax:self.zscale]

        #     ###########################################################
        #     # update rectangle limits from input image coordinates
        #     # this is what the network sees
        #     self.rectangle[i] = Rectangle(xmin, xmax,
        #                                   ymin, ymax,
        #                                   zmin, zmax)
        # return screen

    # Should the argument agent not be renamed to image rather?
    def get_plane(self, z=0, agent=0):
        return self._image[agent].data[:, :, z]

    def _calc_reward(self, current_loc, next_loc):
        """
        Calculate the new reward based on the increase in IOU to
        the target location
        """
        curr_iou = self.calcIou(current_loc, self._target_loc)
        next_iou = self.calcIou(next_loc, self._target_loc)

        if self.reward_method == 'binary':
            return  1 if next_iou > curr_iou else -1


    # TODO: does this not return the oscillation for the first agent only?
    @property
    def _oscillate(self):
        """ Return True if all agents are stuck and oscillating
        """
        for i in range(self.agents):
            counter = Counter(self._loc_history[i])
            freq = counter.most_common()
            # At beginning of episodes, history is prefilled with (0, 0, 0),
            # thus do not count their frequency
            if freq[0][0] == (0, 0, 0):
                if len(freq) < self.oscillations_allowed:
                    return False
                if freq[1][1] < self.oscillations_allowed:
                    return False
            elif freq[0][1] < self.oscillations_allowed:
                return False
        return True

    def get_action_meanings(self):
        """ return array of integers for actions"""
        ACTION_MEANING = {
            1: "UP",  # MOVE Z+
            2: "FORWARD",  # MOVE Y+
            3: "RIGHT",  # MOVE X+
            4: "LEFT",  # MOVE X-
            5: "BACKWARD",  # MOVE Y-
            6: "DOWN",  # MOVE Z-
        }
        return [ACTION_MEANING[i] for i in self.actions]

    @property
    def getScreenDims(self):
        """
        return screen dimensions
        """
        return (self.width, self.height, self.depth)

    def lives(self):
        return None

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)
        # self.num_games = StatCounter()
        # self.num_success = [StatCounter()] * int(self.agents)


class FrameStack(gym.Wrapper):
    """used when not training. wrapper for Medical Env"""
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k  # history length
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape # (90, 90, 90)

        self._base_dim = len(shp)
        new_shape = (k, ) + shp # (1, 4, 90, 90, 90)
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape,
                                            dtype=np.uint8)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        obs = self.env.reset()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self._observation()

    def step(self, action, q_values, done):
        obs, reward, done, info = self.env.step(action, q_values, done)
        # when exceed its max_len, deque will automatially pop from its left
        self.frames.append(obs)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=1)
        # if self._base_dim == 2:
        #     return np.stack(self.frames, axis=-1)
        # else:
        #     return np.concatenate(self.frames, axis=2)

    
def main():
    env = MedicalPlayer()
    env = FrameStack(env, 4)
    curr_state = env.reset()

    fig, ax = plt.subplots()
    ax.imshow(env._box_image[0, 3, :, :, 36],cmap='gray', interpolation=None)

    fig, ax = plt.subplots()
    ax.imshow(curr_state[0, 3, :, :, 22],cmap='gray', interpolation=None)