import torch
import os
import sys
import logging
import numpy as np

from config import config
from dataloader import *
from agent import Agent
from dqn import DQN
from replay_buffer import Replay_Buffer
from medical_env import Medical_Env


def buffer_trajectory(player, env, buffer, qnet_primary, epsilon, episode):
    
    env.reset()




def train():
    return


def main():

    # get training configeration
    device = torch.device(config.device)

    # initialize agent
    player = Agent()

    # initalize the environment
    env = Medical_Env()

    # initialize DQN model
    qnet_primary = DQN().to(device)
    qnet_target = DQN().to(device)
    qnet_target.eval()

    # initialize experience replay buffer
    buffer = Replay_Buffer()

    # load or save model
    save_frequency = int((config.iteration - config.start_iter) / 20)

    total_rewards = []
    q_losses = []

    # start to train the model
    for episode in range(config.start_iter + 1, config.iteration):

        epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
                * np.exp(-1 * (episode % config.switch_iters) / config.epsilon_decay)

        reward, success = buffer_trajectory(player, env, buffer, qnet_primary, epsilon, episode)
        total_rewards.append(reward)

        if buffer.size >= config.batch_size:
            q_loss = train(buffer, qnet_primary, qnet_target, episode)
            q_losses.append(q_loss)

        if episode % config.update_q_target_frequency == 0:
            # qnet_target.target_net_update(q_var, q_target_var)
            qnet_target.target_net_update()

        if episode % config.print_iteration == 0:
            accuracy = buffer.accuracy()
            print('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f'
                % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy))

            logging.info('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f'
                % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy))

            total_rewards = []
            q_losses = []

        if episode % save_frequency == 0:
            # save the model weight
            print('ckpt saved')




if __name__ == "__main__":
    main()