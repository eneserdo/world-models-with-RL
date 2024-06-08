import os
import cv2
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
# import gym
# import gym.envs.box2d
import gymnasium as gym
from utils.misc import RED_SIZE, LSIZE, RSIZE, ASIZE
# A bit dirty: manually change size of car racing env
# gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64


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

    def __init__(self, mdir, record_rgb=False, save_dir="dream"):
        super().__init__()

        self.save_dir = save_dir
        self.record_rgb = record_rgb

        assert torch.cuda.is_available(), "CUDA not available"
        device = 'cuda:0'
        self.env = gym.make('CarRacing-v2')

        self.action_space = self.env.action_space
        self.time_limit = self.env._max_episode_steps

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)


        vae_file, rnn_file = [os.path.join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn']]

        assert os.path.exists(vae_file), f"No trained VAE in the {mdir} directory"
        assert os.path.exists(rnn_file), f"No trained MDRNN in the {mdir} directory" 

        vae_state, rnn_state = [torch.load(fname, map_location={'cuda:0': str(device)}) for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})


        self.mdrnn.eval()
        self.vae.eval()

        self.hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        self.step_count = 0 
        self.z_vec = None
        self.z_vec_next = None

        if record_rgb:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

    def step(self, action):
        """Training step."""
        self.step_count += 1

        mus, sigmas, logpi, r, d, self.next_hidden = self.mdrnn(action, self.z_vec, self.hidden)
        
        self.z_vec_next = get_z_vector(mus, sigmas, logpi)

        action = action.squeeze().cpu().numpy()
        self.hidden = self.next_hidden
        self.z_vec = self.z_vec_next
        reward = r.squeeze().cpu().numpy()
        done = d.squeeze().cpu().numpy()

        if self.record_rgb:
            self._save_img(self.z_vec)

        observation, reward, terminated, truncated, info = self.z_vec, reward, done, self.step_count >= self.time_limit, {}

        return observation, reward, terminated, truncated, info
    
    def _based_step(self, action):
        """Data generation step."""
        self.step_count += 1

        ######### This is the only difference between this and the step method
        obs, _, _, _, _ = self.env.step(action)
        obs = transform(obs).unsqueeze(0).to(self.device)
        _, self.z_vec, _ = self.vae(obs)
        #########

        mus, sigmas, logpi, r, d, self.next_hidden = self.mdrnn(action, self.z_vec, self.hidden)
        
        # self.z_vec_next = get_z_vector(mus, sigmas, logpi)

        action = action.squeeze().cpu().numpy()
        self.hidden = self.next_hidden
        # self.z_vec = self.z_vec_next
        reward = r.squeeze().cpu().numpy()
        done = d.squeeze().cpu().numpy()

        observation, reward, terminated, truncated, info = self.z_vec, reward, done, self.step_count >= self.time_limit, {}

        return observation, reward, terminated, truncated, info



    def reset(self, seed=None, options=None, num_empty_actions=20):
        
        self.hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        
        self.step_count = 0
        obs, _ = self.env.reset()
        
        for _ in range(num_empty_actions):
            obs, _, _, _, _ = self.env.step(np.zeros(3))
        
        obs = transform(obs).unsqueeze(0).to(self.device)
        
        _, z_vec, _ = self.vae(obs)
        # to numpy array
        self.z_vec = z_vec.detach().squeeze().cpu().numpy()
        
        return self.z_vec, {}

    def _save_img(self, sample):
        sample = self.vae.decoder(sample).cpu().numpy()
        img = cv2.resize(img, (None, None), fx=2, fy=2)
        cv2.imwrite(os.path.join(self.save_dir, f"img{self.step_count}"), img)

    def close(self):
        self.env.close()