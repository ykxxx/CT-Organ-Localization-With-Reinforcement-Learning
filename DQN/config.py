from easydict import EasyDict as edict
import torch

config = edict()

# Dataset
config.data_dir = "/Users/kexinyang/Desktop/HDSC 325/Project/data/RawData"
config.dataset = ""
config.batch_size = 4

# Training parameter
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.training = True
config.ad_hoc_structure = False
config.log = True
config.epsilon_exp = True
config.soft_max = False

# Environment
config.max_steps = 50
config.scan_dims = [225, 225, 50]
config.screen_dims = (45, 45, 45)
config.alpha = 0.1
config.tau = 0.85
config.action_step = 10

# DQN Training
config.reward_method = 'binary'
config.min_buffer_experience = 1e4  # min. experiences before start to training
config.learn_every = 3  # no. of experiences between updates to Q_online
config.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
config.lr = 1e-3
config.gamma = 0.5
config.step_size = 50
config.steps_per_episode = 20

config.model_folder =  "/Users/kexinyang/Desktop/HDSC 325/Project/model"
config.model_name = "latest_dqn.pt"
config.start_iter = int(0)
config.num_episode = int(1000)

config.train_frequency = 10
config.save_frequency = 10
config.update_frequency = 10
config.epsilon_min = 0.05
config.epsilon_start = 0.95
config.epsilon_decay = 2