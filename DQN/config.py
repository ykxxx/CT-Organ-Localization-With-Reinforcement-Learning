from easydict import EasyDict as edict
import torch

config = edict()

# Dataset
config.data_dir = "/Users/kexinyang/Desktop/HDSC 325/Project/data"
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
config.alpha = 0.1
config.tau = 0.85

config.exp_folder = './'
config.ckpt_path = 'CKPT'
config.log_dir = './'
config.log_file = 'log1.log'
config.start_iter = int(0)
config.iteration = int(1e6)

config.print_iteration = 1000
config.save_frequency = 10000
config.switch_iters = 200000
config.update_q_target_frequency = 100
config.epsilon_min = 0.05
config.epsilon_start = 0.95
config.epsilon_decay = config.switch_iters / 5
config.gamma = 0.6