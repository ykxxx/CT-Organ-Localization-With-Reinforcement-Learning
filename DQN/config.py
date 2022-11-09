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
config.epsilon_exp = True
config.soft_max = False

# Reply buffer
config.max_memory_size = 20

# Environment
config.max_steps = 50
config.scan_dims = [225, 225, 50]
config.screen_dims = (45, 45, 45)
config.terminate_iou = 0.85
config.action_step_ratio = 0.1
config.min_action_len = 10
config.steps_per_episode = 20
config.history_length = 4
config.num_actions = 11

# DQN Training
config.num_agents = 1
config.reward_method = 'binary'
config.lr = 1e-4
config.step_size = 50
config.gamma = 0.5

config.model_folder =  "/Users/kexinyang/Desktop/HDSC 325/Project/model"
config.model_name = "latest_dqn"
config.log_folder = "/Users/kexinyang/Desktop/HDSC 325/Project/log"
config.log_file = 'log.log'

config.start_iter = int(0)
config.num_episode = int(10e4)
config.train_frequency = 50
config.save_frequency = 100
config.update_frequency = 100
config.epsilon_min = 0.05
config.epsilon_start = 0.95
config.epsilon_decay = 1