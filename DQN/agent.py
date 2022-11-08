import torch
import random
import numpy as np

from config import config
from dqn import DQN
from replay_buffer import ExperienceReplay, Experience

class Agent:
    def __init__(self, mode, image_size, history_len, num_actions, gamma):
        self.mode = mode
        self.image_size = image_size
        self.history_len = history_len
        self.num_actions = num_actions
        self.gamma = gamma

        # Q-Network
        self.qnet_local = DQN(image_size, history_len, num_actions, gamma).to(config.device)
        self.qnet_target = DQN(image_size, history_len, num_actions, gamma).to(config.device)

        # Loss Function
        if config.loss is "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif config.loss is "huber":
            self.loss_fn = torch.nn.HuberLoss(reduction="mean", delta=1.0)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(),lr=config.learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ExperienceReplay(capacity=100000)

        # Training Setting
        self.min_buffer_experience = config.min_buffer_experience  # min. experiences before training
        self.learn_every = config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = config.sync_every  # no. of experiences between Q_target & Q_online sync
        self.batch_size = config.batch_size
        self.device = config.device

        # Save model weight
        self.save_dir = config.save_dir
        self.model_name = config.model_name

    def act(self, state, epsilon):
        """
        Choose an actio based on the input state

        Inputs:
        state (LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action (int): An integer representing which action Mario will perform
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # during training, use epsilon-greedy action selection
        if self.mode == "train":
            if random.random() > epsilon:
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
        # during testing, select the action with the highest q-value
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def store_memory(self, state, next_state, action, reward, done):
        """
        Store the experience to self.replay_buffer

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.replay_buffer.append(Experience(state, action, reward, done, next_state))

    def recall_memory(self):
        """
        Retrieve a batch of experiences from memory
        """
        return self.replay_buffer.sample(self.batch_size)

    def learn(self):
        """
        Update online action value (Q) function with a batch of experiences
        """
        if self.curr_step < self.min_buffer_experience:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        if self.curr_step % self.sync_every == 0:
            self.update_qnet_target()

        if self.curr_step % self.save_every == 0:
            self.save_model_weight()

        # sample from memory
        state, next_state, action, reward, done = self.recall_memory()

        # get q values given current state and action
        current_q = self.get_current_q(state, action)

        # get q values given next state and rewards
        target_q = self.get_target_q(reward, next_state, done)

        # backpropagate loss through target q
        loss = self.fit_qnet_target(current_q, target_q)

        return current_q.mean().item(), loss

    def update_qnet_target(self):
        self.qnet_target.load_state_dict(self.qnet_local.state_dict())

    def fit_qnet_target(self, current_q, target_q):
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_current_q(self, state, action):
        current_Q = self.qnet_local(state)[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def get_target_q(self, reward, next_state, done):
        next_state_q = self.qnet_target(next_state)
        next_q = torch.max(next_state_q, axis=1)

        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def save_model_weight(self):
        save_path = (
            self.save_dir / f"{self.model_name}_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.gamma),
            save_path,
        )
        print(f"{self.model_name} saved to {save_path} at step {self.curr_step}")
        