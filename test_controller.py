""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
from utils.paths import PathResolutionError, resolve_ctrl_file
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

try:
    ctrl_file = resolve_ctrl_file(args.logdir)
except PathResolutionError as exc:
    raise SystemExit(f"error: {exc}")

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000)

with torch.no_grad():
    generator.rollout(None)
