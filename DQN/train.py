import torch
import os
import sys
import logging
import numpy as np
from tqdm import tqdm

from config import config
from dataloader import *
from agents import Agents
from dqn import Network3D
from replay_buffer import ReplayMemory
from medical_env import MedicalPlayer, FrameStack


def set_reproducible():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

def init_buffer(env, agents, num_agents, buffer, epsilon, init_memory_size=None):

    if init_memory_size is None:
        init_memory_size = buffer.max_size

    pbar = tqdm(desc=f"Collect {init_memory_size} experience trajectories in replay buffer", total=init_memory_size)

    while len(buffer) < init_memory_size:

        # reset the environment at the beginning of the episode.
        obs = env.reset()
        terminal = [False for _ in range(num_agents)]
        steps = 0

        for _ in range(config.steps_per_episode):
            steps += 1
            acts, q_values = agents.get_next_actions(obs, epsilon)
            obs, reward, terminal, info = env.step(acts, q_values, terminal)
            buffer.append((obs[:, -1, :, :, :], acts, reward, terminal))

            if all(t for t in terminal):
                break

        pbar.update(steps)

    pbar.close()

    #return buffer

def train(env, agents, num_agents, buffer):
    
    set_reproducible()
        
    init_buffer(env, agents, num_agents, buffer, config.epsilon_start)

    episode = 1
    cum_scores = [0] * num_agents
    cum_iou = [0] * num_agents
    cum_complete_game = [0] * num_agents

    for episode in range(config.num_episode):
        # reset the environment at the beginning of the episode.
        obs = env.reset()
        buffer._hist.clear()
        terminal = [False for _ in range(num_agents)]
        losses = []

        epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
                  * np.exp(-1 * episode / config.num_episode * config.epsilon_decay)

        for step_num in range(config.steps_per_episode):

            # step the agent once, and get the transition tuple
            # index = buffer.append_obs(obs)
            
            acts, q_values = agents.get_next_actions(buffer.recent_state(), epsilon)
            next_obs, reward, terminal, info = env.step(np.copy(acts), q_values, terminal)
            buffer.append((next_obs[:, -1, :, :, :], acts, reward, terminal))

            # buffer.append_effect((index, obs, acts, reward, terminal))
            # obs = next_obs

            cum_scores = [sum(x) for x in zip(cum_scores, reward)]

            if all(t for t in terminal):
                break

        iou_score_list = [info['iou_score_' + str(i)] for i in range(num_agents)]
        complete_game_list = [info['complete_game_' + str(i)] for i in range(num_agents)]
        cum_iou = [sum(x) for x in zip(cum_iou, iou_score_list)]
        cum_complete_game = [sum(x) for x in zip(cum_complete_game, complete_game_list)]
                

        if episode % config.train_frequency == 0:
            
            # states, actions, rewards, next_states, terminal
            mini_batch = buffer.sample(config.batch_size)
            loss = agents.train_q_network(mini_batch, config.gamma)
            losses.append(loss)

            print_score = round(np.mean(cum_scores) / config.train_frequency, 2)
            print_iou = round(np.mean(cum_iou) / config.train_frequency, 3)
            print_terminal = round(np.mean(cum_complete_game) / config.train_frequency, 3)

            print(f"[{episode}] epsilon: {round(epsilon, 3)}, loss: {round(loss, 3)}, score: {print_score}, iou: {print_iou}, complete game: {print_terminal}")

            cum_scores = [0] * num_agents
            cum_iou = [0] * num_agents
            cum_complete_game = [0] * num_agents


        if episode % config.update_frequency == 0:
            agents.copy_to_target_network()

        if episode % config.save_frequency == 0:
            agents.save_model(config.model_name)
            agents.scheduler.step()


def main():

    train_mode = True
    screen_dims = tuple(config.screen_dims)
    max_num_frames = 100
    max_memory_size = 100
    history_length = 4
    num_agents = 1
    num_actions = 11
    action_step = 10
    reward_method = config.reward_method

    env = MedicalPlayer(train_mode, screen_dims, history_length, action_step, max_num_frames, num_agents, reward_method)
    wrapped_env = FrameStack(env, k=4)

    buffer = ReplayMemory(max_memory_size, screen_dims, history_length, num_agents)

    agents = Agents(num_agents, num_actions, history_length, max_memory_size, screen_dims)

    print("-" * 50)
    print("Start training")

    train(wrapped_env, agents, num_agents, buffer)


if __name__ == "__main__":
    main()