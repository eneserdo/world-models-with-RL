""" Training VAE """
import argparse
import os
from os.path import join, exists
from os import mkdir

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')

parser.add_argument('--dataset_dir', type=str, help='Directory where dataset is stored')

parser.add_argument('--verbose', action='store_true', help='chatty')

args = parser.parse_args()
cuda = torch.cuda.is_available()


torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


# TODO: No normalization? Maybe env retuns values in [0, 1]?
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_train = RolloutObservationDataset(args.dataset_dir,
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset(args.dataset_dir,
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)


model = VAE(3, LSIZE, img_size=RED_SIZE).to(device)
optimizer = optim.Adam(model.parameters())


# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


os.makedirs(join(args.logdir, "vae", 'reconstructed'), exist_ok=True)

def save_reconstructed_images(original, recons, epoch_idx):
    # save the original and recontructed images side by side
    N=10
    original_subset = original[:N]
    recons_subset = recons.view(args.batch_size, 3, RED_SIZE, RED_SIZE)[:N]

    comparison = torch.cat([original_subset, recons_subset])
    
    # CHECKME:  # maybe add 0.5 bc save_image expects image in [-0.5, +0.5]
    save_image(comparison.cpu(),
               join(args.logdir, "vae", "reconstructed", 'reconstructed_' + str(epoch_idx) + '.png'), nrow=N)
    
    
train_losses = []
test_losses = []

def vis_losses(train_losses, test_losses, save_path):
    plt.figure()
    plt.title('VAE')
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(save_path, 'vae_losses.png'))
    plt.close()

    np.savez(join(save_path, 'vae_losses.npz'),
             train_losses=train_losses, test_losses=test_losses) 


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        save_reconstructed_images(data, recon_batch, epoch)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            if args.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))


    if args.verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
    
    train_losses.append(train_loss / len(train_loader.dataset))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    if args.verbose:
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    # mkdir(vae_dir)
    os.makedirs(vae_dir)

if not exists(join(vae_dir, 'samples')):
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

def progress_bar(current, total, msg=None):
    """ Progress bar to keep track of training """
    frac = current / total
    filled_progbar = round(frac * 40)
    print('\r', '█' * filled_progbar + '░' * (40 - filled_progbar),
          '[{:>7.2%}]'.format(frac), end='')

    if msg:
        print(' - ' + msg, end='')

    if current == total:
        print()

# TODO: improve training loop
for epoch in range(1, args.epochs + 1):
    progress_bar(epoch, args.epochs, msg='Epoch {}/{}'.format(epoch, args.epochs))
    train(epoch)
    test_loss = test()
    
    test_losses.append(test_loss)

    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            num_of_samples = 64
            sample = torch.randn(num_of_samples, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE), #CHECKME Buna gerek olmamalı?
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break


vis_losses(train_losses, test_losses, join(args.logdir, "vae"))