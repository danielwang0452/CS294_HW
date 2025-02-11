from deepul.hw3_helper import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F4
device = 'mps'
dtype = torch.float32

#visualize_celeb()

class VAE_Encoder(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        N, C, H, W = in_shape
        self.latent_dim = 16
        self.out_linear = nn.Linear(4*4*256, 2*self.latent_dim)
        self.q = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 4x4
            nn.ReLU())

    def forward(self, x):
        out = self.q(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.out_linear(out)
        return out

class VAE_Decoder(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        N, C, H, W = in_shape
        self.latent_dim = 16
        self.p = nn.Sequential( # start with 4x4
# in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1))

        self.in_linear = nn.Linear(self.latent_dim, 4*4*128)

    def forward(self, x):
        out = self.in_linear(x)
        out = F.relu(out)
        N, _ = out.shape
        out = torch.reshape(out, (N, 128, 4, 4))
        print(' ')
        for layer in self.p:
            out = layer(out)
            print(out.shape)
        return out

class VAE(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        N, C, H, W = in_shape
        self.hidden_size = 128
        self.encoder = VAE_Encoder(in_shape)
        self.decoder = VAE_Decoder(in_shape)

    def x_z(self, x):  # q
        # x is shape (N, C, H, W)
        out = self.encoder.forward(x) # (N, 2*latent dim)
        _, n = out.shape
        mu = out[:, :n//2]
        logstd = out[:, n//2:]
        return mu, logstd

    def z_x(self, z):  # p
        out = self.decoder(z) # (N, C, H, W)
        # this should be just the raw pixel values -> MSE recon loss, not NLL
        return out

    def log_prob(self, x):
        z_mu, z_logstd = self.x_z(x) # q
        z_prior = torch.distributions.normal.Normal(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
        z_dist = torch.distributions.normal.Normal(z_mu, z_logstd.exp())
        # reparameterization trick
        z_sample = z_mu + z_logstd.exp() * z_prior.sample(sample_shape=z_mu.shape).to(device)

        # decode
        x_out = self.z_x(z_sample)
        # reconstruction_loss = MSE
        recon_loss_fn = nn.MSELoss()

        recon_loss = recon_loss_fn(x, x_out)
        recon_loss = recon_loss.mean(dim=0).sum()
        #  DKL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) same shape as latent
        DKL = -0.5 * torch.mean(1 + 2 * z_logstd - z_mu.pow(2) - z_logstd.exp().pow(2))
        DKL = DKL.mean(dim=0).sum()

        # reconstruction loss - DKL
        log_px = recon_loss - DKL
        return -log_px, -recon_loss, DKL

    def sample(self, shape, decoder_noise):
        mean = torch.zeros(shape, dtype=dtype).to(device)
        std = torch.ones(shape, dtype=dtype).to(device)
        z = torch.normal(mean, std)
        mu, sigma = self.z_x(z)
        if decoder_noise == True:
            samples = torch.normal(mu, sigma)
        elif decoder_noise == False:
            samples = torch.normal(mu, torch.zeros_like(mu, dtype=dtype).to(device))
        x = model.preprocess(x, reverse=True, dequantize=False)
        x = torch.permute(x, (0, 2, 3, 1))
        return samples

    def preprocess(self, x, reverse=False, dequantize=True):
        if reverse:  # doesn't map back to [0, 4]
            x = 1.0 / (1 + torch.exp(-x))
            x -= 0.05
            x /= 0.9
            return x
        else:
            # dequantization
            if dequantize:
                x += torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(device)
            x /= 4.0

            # logit operation
            x *= 0.9
            x += 0.05
            logit = torch.log(x) - torch.log(1.0 - x)
            return logit

def test(model, dataloader):
    model.eval()
    losses = []
    recon_losses = []
    DKLs = []
    with torch.no_grad():
        for batch in dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)
            x = model.preprocess(x, reverse=False, dequantize=True)

            loss, recon_loss, DKL = model.log_prob(x)
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            DKLs.append(DKL.item())
        print(f'test loss: {loss.item()}')

        return np.array(losses).mean(), np.array(recon_losses).mean(), np.array(DKLs).mean()


def train(train_dataloader, test_dataloader, model, epochs, optimizer):
    losses = []
    recon_losses = []
    DKLs = []
    test_losses = []
    test_recon_losses = []
    test_DKLs = []
    l, r, d = test(model, test_dataloader)
    test_losses.append(l)
    test_recon_losses.append(r)
    test_DKLs.append(d)
    for i in range(epochs):
        for batch in train_dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)
            x = model.preprocess(x, reverse=False, dequantize=True)

            loss, recon_loss, DKL = model.log_prob(x)
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            DKLs.append(DKL.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f'loss: {loss.item()}')

        l, r, d = test(model, test_dataloader)
        test_losses.append(l)
        test_recon_losses.append(r)
        test_DKLs.append(d)

    return np.stack((np.array(losses), np.array(recon_losses), np.array(DKLs)), axis=1), \
        np.stack((np.array(test_losses), np.array(test_recon_losses), np.array(test_DKLs)), axis=1)


def q2_a(train_data, test_data):
  """
  train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
  test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}
  The intended data is (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
   Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255} -> {0, 1, 2, 3}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255} -> {0, 1, 2, 3}
  """
  """ YOUR CODE HERE """
  train_data = np.transpose(train_data, axes=[0, 3, 1, 2]).astype(np.float32)  # NCHW 20000 x 3 x 32 x 32
  test_data = np.transpose(test_data, axes=[0, 3, 1, 2]).astype(np.float32)  # NCHW 6838 x 3 x 32 x 32
  train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
  # model
  model = VAE(in_shape = (64, 3, 32, 32))
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  EPOCHS = 10
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(num_params)

  train_results, test_results = train(train_dataloader, test_dataloader, model, EPOCHS, optimizer)
  samples_1 = model.sample((1000, 2), decoder_noise=True)
  #samples_2 = model.sample((1000, 2), decoder_noise=False)
  print(train_results.shape)
  return train_results, test_results, samples_1.cpu().detach().numpy(), samples_2.cpu().detach().numpy()

q2_save_results_2('a', 1, q2_a)