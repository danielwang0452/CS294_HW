from deepul.hw4_helper import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'mps'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
# timesteps is shape (N), returns shape (N, dim)
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps[:, None].numpy().astype(np.float32) * freqs[None]

    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = np.concatenate([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, temb_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.net1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.SiLU()
    )
    self.net2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.SiLU()
    )
    self.emb_linear = nn.Linear(temb_channels, out_channels)
    self.out_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x, temb):
    h = self.net1(x)
    temb = self.emb_linear(temb)
    h += temb[:, :, None, None] # h is BxDxHxW, temb is BxDx1x1
    h = self.net2(h)
    if self.in_channels != self.out_channels:
        x = self.out_conv(x)
    return x + h

class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)

class Upsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    self.convtranspose = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)

  def forward(self, x):
    #y = F.interpolate(x, scale_factor=2)
    #y = self.conv(y)
    #print(x.shape)
    out = self.convtranspose(x)
    #print(out.shape)
    return out

class UNet(nn.Module):
  def __init__(self, in_channels, hidden_dims, blocks_per_dim):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_dims = hidden_dims
    self.blocks_per_dim = blocks_per_dim
    # networks
    self.temb_channels = self.hidden_dims[0] * 4
    self.emb_net = nn.Sequential(nn.Linear(self.hidden_dims[0], self.temb_channels),
                        nn.SiLU(), nn.Linear(self.temb_channels, self.temb_channels))
    # in_conv
    self.in_conv = nn.Conv2d(self.in_channels, self.hidden_dims[0], 3, padding=1)
    # downsampling resblocks
    prev_ch = self.hidden_dims[0]
    down_block_chans = [prev_ch]
    self.down_resblocks = nn.ModuleList([])
    for i, hidden_dim in enumerate(self.hidden_dims):
      for _ in range(self.blocks_per_dim):
        self.down_resblocks.append(ResidualBlock(prev_ch, hidden_dim, self.temb_channels))
        prev_ch = hidden_dim
        down_block_chans.append(prev_ch)
      if i != len(self.hidden_dims) - 1:
          self.down_resblocks.append(Downsample(prev_ch))
          down_block_chans.append(prev_ch)
    # bottom resblocks
    self.bottom_resblocks = nn.ModuleList(
        [ResidualBlock(prev_ch, prev_ch, self.temb_channels),
         ResidualBlock(prev_ch, prev_ch, self.temb_channels)])
    # upsample resblocks
    self.up_resblocks = nn.ModuleList([])
    for i, hidden_dim in list(enumerate(self.hidden_dims))[::-1]:
        for j in range(self.blocks_per_dim + 1):
            dch = down_block_chans.pop()
            self.up_resblocks.append(ResidualBlock(prev_ch + dch, hidden_dim, self.temb_channels))
            prev_ch = hidden_dim
            if i and j == self.blocks_per_dim:
                self.up_resblocks.append(Upsample(prev_ch))
    # out net
    self.out_net = nn.ModuleList([nn.GroupNorm(num_groups=8, num_channels=prev_ch),
                             nn.SiLU(),
                             nn.Conv2d(prev_ch, self.in_channels, 3, padding=1)])

  def forward(self, x, t):
    # in embed
    emb = torch.tensor(timestep_embedding(t, self.hidden_dims[0]), dtype=dtype).to(device)
    # shape (1, D)
    emb = self.emb_net(emb)
    # in conv
    h = self.in_conv(x)
    # downsample
    hs = [h]
    for layer in self.down_resblocks:
      if isinstance(layer, ResidualBlock):
        h = layer(h, emb)
      if isinstance(layer, Downsample):
        h = layer(h)
      hs.append(h)
    # bottom of UNet
    for layer in self.bottom_resblocks:
      h = layer(h, emb)
    # upsample
    for layer in self.up_resblocks:
      if isinstance(layer, ResidualBlock):
        h = layer(torch.cat((h, hs.pop()), dim=1), emb)
      if isinstance(layer, Upsample):
        h = layer(h)
    # out net
    for layer in self.out_net:
      h = layer(h)
    return h

class ResNet(nn.Module):
    def __init__(self, in_channels, num_blocks, hidden_channels, temb_channels):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        ])
        for i in range(num_blocks):
            self.net.append(ResidualBlock(hidden_channels, hidden_channels, temb_channels))
        self.net.append(nn.Conv2d(hidden_channels, in_channels, 3, padding=1))
        self.temb_channels = temb_channels
        self.emb_net = nn.Sequential(nn.Linear(self.temb_channels, self.temb_channels),
                                     nn.SiLU(), nn.Linear(self.temb_channels, self.temb_channels))

    def forward(self, x, t):
        temb = torch.tensor(timestep_embedding(t, self.temb_channels), dtype=dtype).to(device)
        emb = self.emb_net(temb)
        h = x
        for layer in self.net:
            if isinstance(layer, ResidualBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        return h

class DiffusionModel(nn.Module):
  def __init__(self):
    super().__init__()
    #self.net = ResNet(in_channels=3, num_blocks=6, hidden_channels=256, temb_channels=256)
    self.net = UNet(in_channels=3, hidden_dims=[64, 128, 256], blocks_per_dim=2) # TODO: fill in args
    self.normal_dist = torch.distributions.normal.Normal(torch.tensor((0.0)), torch.tensor((1.0)))
    self.loss_fn = nn.MSELoss()
    # DDPM Sampler
    self.noise_steps=1000
    self.beta_start = 1e-4
    self.beta_end = 0.02
    self.beta = self.prepare_noise_schedule().to(device)
    self.alpha = 1. - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)

  def prepare_noise_schedule(self):
      return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

  def noise_images(self, x, t):
      t.to(device)
      sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
      sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
      Ɛ = torch.randn_like(x)
      return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

  def sample_timesteps(self, n):
      return torch.randint(low=1, high=self.noise_steps, size=(n,))

  def forward(self, x, t): # t is tensor with single element
    x_t, eps = self.noise_images(x, t)
    eps_ = self.net(x_t, t/self.noise_steps)
    return eps, eps_

  def loss(self, x):
    # sample integer timesteps
    t = torch.randint(low=1, high=self.noise_steps, size=(x.shape[0],)) # sample t
    #t = torch.randint(size=(1,), low=1, high=self.noise_steps)
    eps, eps_ = self.forward(x, t)
    loss = self.loss_fn(eps, eps_)
    return loss

  def sample(self, n_samples): # sample n steps
    self.eval()
    with torch.no_grad():
        x = self.normal_dist.sample((n_samples, 3, 32, 32)).to(device)
        for i in reversed(range(1, self.noise_steps)):
            t = (i/self.noise_steps)*torch.ones((n_samples), dtype=dtype)
            eps_hat = self.net(x, t)
            t = (torch.ones(n_samples) * i).long().to(device)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * eps_hat) + torch.sqrt(beta) * noise
        return x.permute((0, 2, 3, 1))

def train(model, train_dataloader, test_dataloader, optimizer, epochs):
  def test(model, test_dataloader):
    losses = []
    model.eval()
    with torch.no_grad():
      for batch in test_dataloader:
        x = torch.tensor(batch, dtype=dtype).to(device)
        loss = model.loss(x)
        losses.append(loss.item())
        #break
      test_loss = np.array(losses).mean()
      print(test_loss)
      return test_loss

  train_losses = []
  test_losses = []
  test_losses.append(test(model, test_dataloader))
  for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
      x = torch.tensor(batch, dtype=dtype).to(device)
      loss = model.loss(x)
      train_losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #print(loss.item())
      #break
    test_losses.append(test(model, test_dataloader))
  return np.array(train_losses), np.array(test_losses)

def q2(train_data, test_data):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(2, np.linspace(0, 9, 10)).astype(int)
    """

    """ YOUR CODE HERE """
    # normalise dataset
    train_losses = np.array([1.0])
    test_losses = np.array([1.0])
    train_data = 2*(train_data - 0.5).transpose((0, 3, 1, 2))
    test_data = 2*(test_data - 0.5).transpose((0, 3, 1, 2))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # define model & hyperparams
    model = DiffusionModel().to(device)
    epochs = 2
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # load model
    state_dict = torch.load('DDPM_S1.pth', map_location='mps')
    model.load_state_dict(state_dict)
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, epochs=2)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=3e-4)
    #train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer2, epochs=60)

    #train_losses = np.concatenate((train_losses, train_losses2), axis=0)
    #test_losses = np.concatenate((test_losses, test_losses2), axis=0)

    # save model
    model_path = 'DDPM_S2.pth'
    torch.save(model.state_dict(), model_path)
    # sample
    samples =  model.sample(10).unsqueeze(0)
    for sample_steps in np.power(2, np.linspace(0, 9, 10)).astype(int):
      if sample_steps > 1:
        sample = model.sample(10).unsqueeze(0)
        samples = torch.cat((samples, sample), dim=0)
    samples = samples.cpu().detach().numpy()
    samples = samples/2 + 0.5
    # unnormalise data
    return train_losses, test_losses, samples

q2_save_results(q2)