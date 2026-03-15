""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")

parser.add_argument('--dataset_dir', type=str, help='Directory where dataset is stored')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# assert args.include_reward, "Reward is not included. Are you sure?"

# constants
BSIZE = 16
SEQ_LEN = 32
epochs = 100

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LSIZE, img_size=96).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
# optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

# transform = transforms.Compose([
#     transform_,
#     transforms.ToPILImage(),
#     transforms.Resize((RED_SIZE, RED_SIZE)),
#     transforms.ToTensor(),
# ])

train_loader = DataLoader(
    RolloutSequenceDataset(args.dataset_dir, SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=True, drop_last=True)
test_loader = DataLoader(
    RolloutSequenceDataset(args.dataset_dir, SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8,  drop_last=True)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        # Since we are working with original size now, no need to resize
        # TODO: make it parametric
        # obs, next_obs = [
        #     f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
        #                mode='bilinear', align_corners=True)
        #     for x in (obs, next_obs)]
        obs = obs.view(-1, 3, SIZE, SIZE)
        next_obs = next_obs.view(-1, 3, SIZE, SIZE)

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs

def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

def save_losses():
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(train_losses['loss'], label='train')
    plt.plot(test_losses['loss'], label='test')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_losses['bce'], label='train')
    plt.plot(test_losses['bce'], label='test')
    plt.title('BCE')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_losses['gmm'], label='train')
    plt.plot(test_losses['gmm'], label='test')
    plt.title('GMM')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(train_losses['mse'], label='train')
    plt.plot(test_losses['mse'], label='test')
    plt.title('MSE')
    plt.legend()

    plt.tight_layout()

    save_dir = join(args.logdir, "mdrnn")

    plt.savefig(join(save_dir, 'mdnrnn_loss_curve.png'))

    # Save the losses
    np.savez(join(save_dir, 'mdnrnn_losses.npz'),
            train_bce= np.array(train_losses['bce']),
            train_gmm=np.array(train_losses['gmm']),
            train_mse=np.array(train_losses['mse']),
            train_loss=np.array(train_losses['loss']),
            test_bce=np.array(test_losses['bce']),
            test_gmm=np.array(test_losses['gmm']),
            test_mse=np.array(test_losses['mse']),
            test_loss=np.array(test_losses['loss']))

def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()

            torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), 1)

            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()

    cum_loss = cum_loss * BSIZE / len(loader.dataset)
    cum_bce = cum_bce * BSIZE / len(loader.dataset)
    cum_gmm = cum_gmm * BSIZE / len(loader.dataset) / LSIZE
    cum_mse = cum_mse * BSIZE / len(loader.dataset)

    if train:
        train_losses['loss'].append(cum_loss)
        train_losses['bce'].append(cum_bce)
        train_losses['gmm'].append(cum_gmm)
        train_losses['mse'].append(cum_mse)
    else:
        test_losses['loss'].append(cum_loss)
        test_losses['bce'].append(cum_bce)
        test_losses['gmm'].append(cum_gmm)
        test_losses['mse'].append(cum_mse)

    return cum_loss


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
train_losses = {"bce": [], "gmm": [], "mse": [], "loss": []}
test_losses = {"bce": [], "gmm": [], "mse": [], "loss": []}

for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break

    save_losses()