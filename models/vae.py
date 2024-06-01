
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

# # # # # # # # #
# # # # Encoder-Decoder for 64x64 and 160x160 images with norm layers
# # # # # # # # #

class EnlargedDecoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, image_size):
        super(EnlargedDecoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        if image_size == 96:
            out_f1 = 256
            out_x = 1
        elif image_size == 160:
            out_f1 = 256*3*3
            out_x = 3

        self.fc1 = nn.Linear(latent_size, out_f1)
        # self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

        layers = [
            nn.ConvTranspose2d(out_f1, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, img_channels, 4, stride=2),
            nn.Sigmoid()
        ]

        self.net = nn.Sequential(*layers)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        # x = F.relu(self.deconv1(x))
        # x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3(x))

        # CHECKME: why sigmoid? not tanh? is input [0-1] or [-1, 1]?
        # reconstruction = F.sigmoid(self.deconv4(x))

        reconstruction = self.net(x)

        return reconstruction

class EnlargedEncoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, img_size):
        super(EnlargedEncoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        # self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        # self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        layers = [
            nn.Conv2d(img_channels, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        ]

        last = []
        if img_size == 160:
            last.append(nn.Conv2d(256, 256, 4, stride=2))
            last.append(nn.BatchNorm2d(256))
            last.append(nn.LeakyReLU())
        
        elif img_size == 96:
            last.append(nn.Conv2d(256, 256, 4, stride=1))
            last.append(nn.BatchNorm2d(256))
            last.append(nn.LeakyReLU())



        self.net = nn.Sequential(*layers)
        self.net_last = nn.Sequential(*last)
        

        if img_size == 96:
            self.fc_mu = nn.Linear(1*1*256, latent_size)
            self.fc_logsigma = nn.Linear(1*1*256, latent_size)

        elif img_size == 160:
            self.fc_mu = nn.Linear(3*3*256, latent_size)
            self.fc_logsigma = nn.Linear(3*3*256, latent_size)


    def forward(self, x):
        x = self.net(x)
        x = self.net_last(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma



class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size, img_size=64):
        super(VAE, self).__init__()

        if img_size == 64:
            self.encoder = Encoder(img_channels, latent_size)
            self.decoder = Decoder(img_channels, latent_size)
        elif img_size == 96:
            self.encoder = EnlargedEncoder(img_channels, latent_size, img_size)
            self.decoder = EnlargedDecoder(img_channels, latent_size, img_size)
        elif img_size == 160:
            self.encoder = EnlargedEncoder(img_channels, latent_size, img_size)
            self.decoder = EnlargedDecoder(img_channels, latent_size, img_size)
        else:
            raise ValueError("Invalid image size")


    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
