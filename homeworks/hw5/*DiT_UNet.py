from deepul.hw4_helper import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'mps'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

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
  def __init__(self, in_channels, hidden_dims, blocks_per_dim, num_classes):
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
    self.embedding = nn.Embedding(num_classes + 1, self.hidden_dims[0])

  def forward(self, x, labels, t):
    # in embed
    emb = torch.tensor(timestep_embedding(t, self.hidden_dims[0]), dtype=dtype).to(device) \
      + self.embedding(labels.to(device))
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

class DiffusionModel(nn.Module):
  def __init__(self, vae, scale_factor):
    super().__init__()
    #self.net = ResNet(in_channels=3, num_blocks=6, hidden_channels=256, temb_channels=256)
    self.scale_factor = scale_factor
    self.vae = vae.eval()
    self.net = UNet(in_channels=4, hidden_dims=[128, 256], blocks_per_dim=2, num_classes=10) # TODO: fill in args
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
      sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
      sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
      Ɛ = torch.randn_like(x)
      return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

  def sample_timesteps(self, n):
      return torch.randint(low=1, high=self.noise_steps, size=(n,))

  def forward(self, x, t, labels): # t is tensor with single element
    # encode images
    x_latent = self.vae.encode(x)/self.scale_factor
    x_t, eps = self.noise_images(x_latent, t)
    eps_ = self.net(x_t, labels, t/self.noise_steps)
    return eps, eps_

  def dropout_classes(self, labels, cfg_dropout_prob=0.1):
      labels[:int(len(labels)*cfg_dropout_prob)] = 10
      return labels

  def loss(self, x, labels): # labels are (N,)
    # sample integer timesteps
    t = torch.randint(low=1, high=self.noise_steps, size=(x.shape[0],)) # sample t
    # drop class label
    labels = self.dropout_classes(labels, cfg_dropout_prob=0.1) # Randomly dropout to train unconditional image generation
    eps, eps_ = self.forward(x, t, labels)
    loss = self.loss_fn(eps, eps_)
    return loss

  def sample(self, n_samples, target_class): # sample n steps
    self.eval()
    target_class = target_class*torch.ones((n_samples), dtype=torch.int)
    with torch.no_grad():
        x = self.normal_dist.sample((n_samples, 4, 8, 8)).to(device)
        for i in reversed(range(1, self.noise_steps//1)):
            t = (i/self.noise_steps)*torch.ones((n_samples), dtype=dtype)
            eps_hat = self.net(x, target_class, t)
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
        # decode
        x = self.vae.decode(x*self.scale_factor)
        return x.permute((0, 2, 3, 1))

def train(model, train_dataloader, test_dataloader, optimizer, epochs):
  def test(model, test_dataloader):
    losses = []
    model.eval()
    with torch.no_grad():
      for batch in test_dataloader:
        image, label = batch
        x = torch.tensor(image, dtype=dtype).to(device)
        loss = model.loss(x, label)
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
      image, label = batch
      x = torch.tensor(image, dtype=dtype).to(device)
      loss = model.loss(x, label)
      train_losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #print(loss.item())
      #break
    test_losses.append(test(model, test_dataloader))
  return np.array(train_losses), np.array(test_losses)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def compute_scale_factor(images, vae):
    # reconstruct images
    vae.to(device)
    images_in = images[:1000]
    '''
    images_in = 2*(images_in.transpose((0, 3, 1, 2))-0.5)
    encoded_images = vae.encode(images_in) # shape (N, 4, 8, 8)
    decoded_images = vae.decode(encoded_images) # shape (N, 3, 32, 32)
    decoded_images = decoded_images.cpu().numpy().transpose((0, 2, 3, 1))/2+0.5
    decoded_images = np.expand_dims(decoded_images, 1)
    autoencoded_images = np.concatenate((np.expand_dims(images[:50], 1), decoded_images), axis=1)
    '''
    # compute scale factor
    images_in = 2 * (images_in.transpose((0, 3, 1, 2)) - 0.5)
    encoded_images = vae.encode(images_in)  # shape (N, 4, 8, 8)
    scale_factor = np.std(encoded_images.cpu().numpy().flatten())
    return scale_factor

def q3_b(train_data, train_labels, test_data, test_labels, vae):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    train_labels: A (50000,) numpy array of class labels
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]
    test_labels: A (10000,) numpy array of class labels
    vae: a pretrained VAE

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific class (i.e. row 0 is class 0, row 1 class 1, ...). Use 512 diffusion timesteps
    """
    """ YOUR CODE HERE """

    # normalise dataset
    scale_factor = compute_scale_factor(train_data, vae)
    train_losses = np.array([1.0])
    test_losses = np.array([1.0])
    train_data = 2*(train_data - 0.5).transpose((0, 3, 1, 2))
    test_data = 2*(test_data - 0.5).transpose((0, 3, 1, 2))
    train_dataset = Dataset(train_data, train_labels)
    test_dataset = Dataset(test_data, test_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    # define model & hyperparams
    model = DiffusionModel(vae, scale_factor).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # load model
    state_dict = torch.load('DiT_UNet_2.pth', map_location='mps')
    model.load_state_dict(state_dict)
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, epochs=40)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=5e-4)
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer2, epochs=40)

    #train_losses = np.concatenate((train_losses, train_losses2), axis=0)
    #test_losses = np.concatenate((test_losses, test_losses2), axis=0)

    # save model
    model_path = 'DiT_UNet_2.pth'
    torch.save(model.state_dict(), model_path)
    # sample
    print('sampling')
    samples = model.sample(10, 0).unsqueeze(0)
    for target_class in range(9):
      if target_class > 0:
        sample = model.sample(10, target_class+1).unsqueeze(0)
        samples = torch.cat((samples, sample), dim=0)
    samples = samples.cpu().detach().numpy()
    samples = samples/2 + 0.5
    # unnormalise data
    return train_losses, test_losses, samples

q3b_save_results(q3_b)