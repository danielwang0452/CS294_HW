from deepul.hw4_helper import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'mps'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MultiHeadAttention(nn.Module):
  def __init__(self, hidden_size, num_heads):
    super().__init__()
    self.d_m = hidden_size
    self.d_k = hidden_size
    self.d_v = hidden_size
    self.h = num_heads
    self.k_linear = nn.Linear(self.d_m, self.h*self.d_k)
    self.q_linear = nn.Linear(self.d_m, self.h*self.d_k)
    self.v_linear = nn.Linear(self.d_m, self.h*self.d_v)
    self.out_linear = nn.Linear(self.h*self.d_v, self.d_m)

  def forward(self, x): # x is shape (N, L, d_m)
    N, L, _ = x.shape
    # project to h*d_k (N, L, h*d_k) and reshape to (N, h, L, d_k)
    Q = self.q_linear(x).reshape(N, L, self.h, self.d_k).permute((0, 2, 1, 3))
    K = self.k_linear(x).reshape(N, L, self.h, self.d_k).permute((0, 2, 1, 3))
    V = self.v_linear(x).reshape(N, L, self.h, self.d_v).permute((0, 2, 1, 3))
    attention = F.softmax(torch.matmul(Q, torch.transpose(K, 2, 3))*(self.d_k**(-1/2)), dim=-1)
    attention = torch.matmul(attention, V)
    attention = attention.permute((0, 2, 1, 3)).reshape(N, L, self.h*self.d_v)
    # attention is shape (N, L, h*d_v)
    out = self.out_linear(attention)
    return out

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size)
        )
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.MLP = nn.Sequential(
        nn.Linear(hidden_size, 4*hidden_size),
        nn.GELU(),
        nn.Linear(4*hidden_size, hidden_size))

    def forward(self, x, c):
        # x is (N, L, D) c is (N, D)
        # conditioning
        c = self.cond_mlp(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        # adaLN
        h = self.layernorm(x)
        h = modulate(h, shift_msa, scale_msa)
        # attention
        x = x + gate_msa.unsqueeze(1) * self.attention(h)
        # adaLN
        h = self.layernorm(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.MLP(h)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.cond_mlp = self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size)
        )
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.out_linear = nn.Linear(hidden_size, patch_size*patch_size*out_channels)

    def forward(self, x, c):
        # x is (N, L, D) c is (N, D)
        # conditioning
        c = self.cond_mlp(c)
        shift, scale = c.chunk(2, dim=1)
        # adaLN
        x = self.layernorm(x)
        x = modulate(x, shift, scale)
        x = self.out_linear(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps[:, None].cpu().numpy() * freqs[None]

    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = np.concatenate([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class DiT(nn.Module):
    def __init__(self, input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes):
        super().__init__()
        C, H, W = input_shape
        self.p = patch_size
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            self.blocks.append(DiTBlock(hidden_size, num_heads))
        self.finallayer = FinalLayer(hidden_size, patch_size, out_channels=C)
        self.patch = nn.Conv2d(C, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.embedding = nn.Embedding(num_classes+1, hidden_size)
        self.hidden_size = hidden_size
        self.out_channels = C
        self.pos_embed = None
        #self.unpatch = nn.ConvTranspose2d(, in_channels, 4, 2, 1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.p
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unpatchify2(self, patches, patch_size, img_height, img_width):
        N, L, D = patches.shape  # (128, 16, 192)
        H, W = img_height, img_width
        C = D // (patch_size * patch_size)
        #'''
        h_patches = img_height // patch_size
        w_patches = img_width // patch_size
        # Reshape patches to (B, H_patches, W_patches, patch_size, patch_size, C)
        patches = patches.view(N, h_patches, w_patches, patch_size, patch_size, C)
        # Permute to (B, C, H_patches * patch_size, W_patches * patch_size)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        # Reshape to (B, C, H, W)
        img = patches.view(N, C, img_height, img_width)
        #'''
        fold = nn.Fold(output_size=(H, W), kernel_size=self.p, stride=self.p)
        #img = fold(patches.permute(0, 2, 1))
        return img

    def forward(self, x, t):
        N, C, H, W = x.shape
        # image, label, diffusion timestep
        # x is (N, C, H, W) y is (N) t is (N)
        # patchify and flatten with strided conv
        # B x C x H x W -> B D H/P W/P -> B x (H // P * W // P) x D
        x = self.patch(x).reshape((N, self.hidden_size, H*W//(self.p**2))).permute(0, 2, 1)
        # positional embedding
        if self.pos_embed == None:
            self.pos_embed = torch.tensor(
                get_2d_sincos_pos_embed(embed_dim=x.shape[2], grid_size=H // self.p),
                dtype=dtype).to(device)
        x += self.pos_embed
        # conditioning - t and y should both be (N,)
       #y = self.embedding(y.to(device))
        t = torch.tensor(timestep_embedding(t, self.hidden_size), dtype=dtype).to(device)
        c = t
        # layers
        for layer in self.blocks: # DiT blocks
            x = layer(x, c)
        out = self.finallayer(x, c)
        # unpatchify: B x (H // P * W // P) x (P * P * C) -> B x C x H x W
        out = self.unpatchify(out)#, self.p, H, W)
        return out

class DiffusionModel(nn.Module):
  def __init__(self):
    super().__init__()
    #self.net = ResNet(in_channels=3, num_blocks=6, hidden_channels=256, temb_channels=256)
    self.net = DiT(
        input_shape=(3, 32, 32),
        patch_size=4,
        hidden_size=256, # 512
        num_heads=4, # 8
        num_layers=6, # 12
        num_classes=10)
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
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # load model
    #state_dict = torch.load('DDPM_DiT_1.pth', map_location='mps')
    #model.load_state_dict(state_dict)
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, epochs=10)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=3e-4)
    #train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer2, epochs=3)

    #train_losses = np.concatenate((train_losses, train_losses2), axis=0)
    #test_losses = np.concatenate((test_losses, test_losses2), axis=0)

    # save model
    model_path = 'DDPM_DiT_1.pth'
    torch.save(model.state_dict(), model_path)
    # sample
    print('sampling')
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