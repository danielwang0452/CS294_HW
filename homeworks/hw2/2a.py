from deepul.hw2_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepul.pytorch_util as ptu
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

device = 'mps'
dtype = torch.float32

#------------------------------------------------------------------------

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, batch_size):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.mask_type = mask_type
        self.padding = padding
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.create_mask()

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        return out

    def create_mask(self):
        # weight is shape (out_channels, in_channels, H, W)
        self.mask = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.mask[:, :, :self.kernel_size//2, :] = 1
        self.mask[:, :, self.kernel_size//2, :self.kernel_size//2] = 1
        if self.mask_type == 'B':
            self.mask[:, :, self.kernel_size // 2, self.kernel_size // 2] = 1
        self.mask = torch.tensor(self.mask, dtype=torch.float32).to(device)

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape[1])
    # layernorm only applied across the 64 channels
    def forward(self, x):
        # x is shape (N, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        return out.permute(0, 3, 1, 2).contiguous()

class ResBlock(nn.Module):
    def __init__(self, mid_kernel_size, in_channels, batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.mid_kernel_size = mid_kernel_size
        self.batch_size = batch_size
        self.block = nn.ModuleList([MaskedConv('B',
                                      in_channels=self.in_channels,
                                      out_channels=self.in_channels//2,
                                      kernel_size=1,
                                      padding=0,
                                      batch_size=self.batch_size),
                                    nn.ReLU(),
                                    MaskedConv('B',
                                      in_channels=self.in_channels//2,
                                      out_channels=self.in_channels // 2,
                                      kernel_size=self.mid_kernel_size,
                                      padding=self.mid_kernel_size//2,
                                      batch_size=self.batch_size),
                                    nn.ReLU(),
                                    MaskedConv('B',
                                      in_channels=self.in_channels // 2,
                                      out_channels=self.in_channels,
                                      kernel_size=1,
                                      padding=0,
                                      batch_size=self.batch_size),
                                    nn.ReLU()
                                    ],
                                      )

    def forward(self, x):
        # x is shape (N, C, H, W)
        out = x
        for layer in self.block:
            out = layer(out)
        return out + x

class PixelCNN(nn.Module):
  def __init__(self, num_blocks, num_filters, filter_size, batch_size, image_dim, output_channels):
    super().__init__()
    self.num_blocks = num_blocks
    self.num_filters = num_filters # num filters in each layer = output channels
    self.filter_size = filter_size # should be an odd integer
    self.batch_size = batch_size
    self.in_channels, self.iH, self.iW = image_dim
    self.output_channels = output_channels # this is for output layer
    # build network
    # input layer
    model = nn.ModuleList([MaskedConv('A',
                                      in_channels=self.in_channels,
                                      out_channels=self.num_filters,
                                      kernel_size=self.filter_size,
                                      padding=self.filter_size//2,
                                      batch_size=self.batch_size),
                           nn.ReLU()])
    # resblocks
    for i in range(self.num_blocks):
        # resblock __init__(self, mid_kernel_size, in_channels, batch_size):
        model.append(ResBlock(7, self.num_filters, self.batch_size))
        model.append(LayerNorm((self.batch_size, self.num_filters, self.iH, self.iW)))
    # output layer
    model.append(MaskedConv('A',
               in_channels=self.num_filters,
               out_channels=self.output_channels,
               kernel_size=self.filter_size,
               padding=self.filter_size // 2,
               batch_size=self.batch_size))
    self.net = model

  def forward(self, x):
    '''
    :param x: shape (batch size, in channels, iH, iW)
    :return: shape (batch size, output channels, iH, iW)
    '''
    out = x.to(device)
    a = nn.ReLU()
    for l, layer in enumerate(self.net):
        out = layer(out)
    return out

class PixelCNNAutoregressiveFlow(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.PixelCNN = PixelCNN(num_blocks=3,
                                 num_filters=64,
                                 filter_size=7,
                                 batch_size=128,
                                 image_dim=[1, 20, 20],
                                 output_channels=n_components*3)
        self.mix_dist = torch.distributions.normal.Normal
        self.base_dist = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))

    def forward(self, x):
        # x is shape (N, C, H, W)
        loc, scale, weight = torch.chunk(self.PixelCNN(x), chunks=3, dim=1)
        # loc, scale, weight are (N, n_components, H, W)
        scale_weight = F.softmax(weight, dim=1)
        z = (self.mix_dist(loc, scale.exp()).cdf(
            x.repeat(1, self.n_components, 1, 1)) * scale_weight).sum(dim=1) # weight across components
        # z is (N, H, W)

        log_det = (self.mix_dist(loc, scale.exp()).log_prob(
            x.repeat(1, self.n_components, 1, 1)).exp() * scale_weight).sum(dim=(1)).log() # weight across components
        # log_det is (N, H, W)
        return z, log_det

    def invert(self, x, z, c, h, w):
        # z shape = (N, n_components, H, W)
        loc, scale, weight = torch.chunk(self.PixelCNN(x), chunks=3, dim=1)
        # each of shape (N, n_components, H, W)
        # loc, scale, weight are (N, n_components, H, W)
        scale_weight = F.softmax(weight, dim=1)
        # now invert
        x = (self.mix_dist(loc, scale.exp()).icdf(z*scale_weight)).sum(dim=1)
        # shape (N, H, W)
        return x

    def log_prob(self, x):
        z, z_det = self.forward(x) # shape (N, H, W)
        return (self.base_dist.log_prob(z) + z_det).mean(dim=(1, 2))  # avg across image size

    def loss(self, x):
        return -self.log_prob(x).mean() # avg across batch

    def sample(self, n_samples, C, H, W, n_components):
        shape = (n_samples, n_components, H, W)
        samples = np.zeros((n_samples, H, W, C))
        self.eval()
        with torch.no_grad():
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        x = torch.tensor(samples, dtype=dtype).permute((0, 3, 1, 2)).to(device)
                        z = torch.normal(mean=torch.zeros(shape, dtype=dtype),
                                         std=torch.ones(shape, dtype=dtype)).to(device)
                        x_sample = self.invert(x, z, C, H, W)
                        samples[:, h, w, c] = x_sample[:, h, w].cpu().numpy()
                print(h)
        return samples

def train(model, train_dataloader, test_dataloader, epochs, optimizer):
    train_losses = []
    test_losses = []
    test_losses.append(test(model, test_dataloader))
    for epoch in range(epochs):
        model.train()
        # print(epoch)
        for batch in train_dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)
            loss = model.loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        print(f'train loss: {loss}')
        test_losses.append(test(model, test_dataloader))
    return np.array(train_losses), np.array(test_losses)

def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        epoch_losses = []
        for batch in dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)
            loss = model.loss(x)
            epoch_losses.append(loss.item())
        loss = np.array(epoch_losses).mean()
        print(f'test loss: {loss}')
        return loss

def q2(train_data, test_data):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    H = W = 20
    Note that you should dequantize your train and test data, your dequantized pixels should all lie in [0,1]

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in [0, 1], where [0,0.5] represents a black pixel
        and [0.5,1] represents a white pixel. We will show your samples with and without noise.
    """

    """ YOUR CODE HERE """
    _, H, W, C = train_data.shape
    # data prepreocessing
    train_data = np.transpose(train_data, (0, 3, 1, 2)).astype(np.float32) / 2.0
    test_data = np.transpose(test_data, (0, 3, 1, 2)).astype(np.float32) / 2.0
    # shape(N, C, H, W), then add noise to dequantize data
    train_data = train_data + np.random.uniform(high=0.5, size=train_data.shape)
    test_data = test_data + np.random.uniform(high=0.5, size=test_data.shape)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # model
    n_components = 10
    model = PixelCNNAutoregressiveFlow(n_components)
    model.to(device)
    EPOCHS = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # train
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, EPOCHS, optimizer)
    samples = model.sample(100, C, H, W, n_components)
    return train_losses, test_losses, samples
q2_save_results(q2)