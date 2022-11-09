import os
import torch
import random
import numpy as np
from tqdm import tqdm

from config import config
from dqn import Network3D
from replay_buffer import ReplayMemory


class Agents():
    def __init__(self, num_agents, num_actions, hist_len, max_memory_size, state_shape):

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.hist_len = hist_len
        self.max_memory_size = max_memory_size
        self.state_shape = state_shape      

        # q networks
        self.q_network = Network3D(self.num_agents, self.hist_len, self.num_actions).float().to(config.device)
        self.target_network = Network3D(self.num_agents, self.hist_len, self.num_actions).float().to(config.device)

        self.copy_to_target_network()

        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
            p.requires_grad = False

        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.step_size, gamma=config.gamma)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, model_name="dqn.pt"):

        if not os.path.exists(config.log_folder):
            os.makedirs(config.model_folder)

        model_path = os.path.join(config.model_folder, model_name)
        torch.save(self.q_network.state_dict(), model_path)

    # Function that is called whenever we want to train the Q-network. Each
    # call to this function takes in a transition tuple containing the data we
    # use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimizer.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the
        # loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimizer.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape
        (states, actions, rewards, next_states, dones)
        '''
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)

        y = self.target_network.forward(next_state)
        # dim (batch_size, agents, num_actions)
        y = y.view(-1, self.num_agents, self.num_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]
        # dim (batch_size, agents, num_actions)
        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.num_agents, self.num_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)
        # Bellman equation
        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        # td_errors = (network_prediction -
        # batch_labels_tensor.unsqueeze(-1)).detach() # TODO td error needed
        # for exp replay

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = torch.gather(network_prediction, -1, actions).squeeze()

        # Update transitions' weights
        # self.buffer.recompute_weights(transitions, td_errors)

        return self.loss_fn(batch_labels_tensor.flatten(), y_pred.flatten())

    def get_greedy_actions(self, obs_stack, doubleLearning=True):
        inputs = torch.tensor(obs_stack).unsqueeze(0)

        # print(inputs.shape)
        if doubleLearning:
            q_vals = self.q_network.forward(inputs).detach().squeeze(0)
        else:
            q_vals = self.target_network.forward(
                inputs).detach().squeeze(0)
        idx = torch.max(q_vals, -1)[1]
        greedy_steps = np.array(idx, dtype=np.int32).flatten()

        return greedy_steps, q_vals.data.numpy()

    def get_next_actions(self, obs_stack, epsilon):
        # epsilon-greedy policy
        if np.random.random() < epsilon:
            q_values = np.zeros((self.num_agents, self.num_actions))
            actions = np.random.randint(self.num_actions, size=self.num_agents)
        else:
            actions, q_values = self.get_greedy_actions(
                obs_stack, doubleLearning=True)

        return actions, q_values
    

def main():
    max_size = 100
    hisotry_len = 4
    num_agents = 1
    num_actions = 6
    state_shape = tuple(config.screen_dims)

    buffer = ReplayMemory(max_size, state_shape, hisotry_len, num_agents)

    agents = Agents(num_agents, num_actions, hisotry_len, max_size, state_shape)


if __name__ == "__main__":
    main()