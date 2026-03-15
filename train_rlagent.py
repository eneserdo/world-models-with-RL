"""
Train RL agent inside the World Models framework
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

import stable_baselines3 as sb3
from stable_baselines3 import PPO

from dream_wrap import CustomEnv

train = True

# parsing
# parser = argparse.ArgumentParser()
# parser.add_argument('--logdir', type=str, help='Where everything is stored.')

# args = parser.parse_args()

# dreamland = RolloutGenerator(args.logdir)

# Create the environment
env = CustomEnv(mdir="exp_dir_car", device='cuda:0', time_limit=100000)

if train:
    # # Initialize the agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the agent
    model.save("ppo_dummy_world_model")
else:
    model = PPO.load("ppo_dummy_world_model")
    
# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()