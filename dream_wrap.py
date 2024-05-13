import gymnasium as gym
import numpy as np
from gymnasium import spaces


import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
# import gym
# import gym.envs.box2d
import gymnasium as gym


# A bit dirty: manually change size of car racing env
# gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Action, latent, recurrent
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb

def get_z_vector(mus, sigmas, logpi):
    """Sample a latent vector from a MDN."""
    # select the component
    pi = torch.exp(logpi)
    m = torch.distributions.Categorical(pi)
    sel = m.sample().view(-1, 1, 1)
    # get the z corresponding to the component
    mus = mus.view(-1, mus.size(-1))
    mus = mus.gather(0, sel.expand(mus.size(1), -1).t()).t().view(-1, 1, mus.size(-1))
    sigmas = sigmas.view(-1, sigmas.size(-1))
    sigmas = sigmas.gather(0, sel.expand(sigmas.size(1), -1).t()).t().view(-1, 1, sigmas.size(-1))
    # reparametrize
    z_vec = mus + sigmas * torch.randn_like(sigmas)
    return z_vec

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, mdir, device, time_limit):
        super().__init__()

        self.env = gym.make('CarRacing-v2')
        # # Define action and observation space
        # # They must be gym.spaces objects
        # # Example when using discrete actions:
        self.action_space = self.env.action_space
        # # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = self.env.observation_space

        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.device = device

        self.time_limit = time_limit

        self.hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        self.step_count = 0 

    def step(self, action):
        self.step_count += 1

        mus, sigmas, logpi, r, d, next_hidden = self.mdrnn(action, z_vec, hidden)
        
        z_vec_next = get_z_vector(mus, sigmas, logpi)

        action = action.squeeze().cpu().numpy()
        hidden = next_hidden
        z_vec = z_vec_next
        reward = r.squeeze().cpu().numpy()
        done = d.squeeze().cpu().numpy()

        observation, reward, terminated, truncated, info = z_vec, reward, done, self.step_count >= self.time_limit, {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset()
        obs = transform(obs).unsqueeze(0).to(self.device)
        _, z_vec, _ = self.vae(obs)
        return z_vec, info

    def render(self):
        sample = self.vae.decoder(sample).cpu()
        sample = sample.view(64, 3, RED_SIZE, RED_SIZE)
        # TODO: render sample 
         

    def close(self):
        self.env.close()