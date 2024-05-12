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

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')

args = parser.parse_args()

dreamland = RolloutGenerator(args.logdir)
