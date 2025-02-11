from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'mps'
'''
m = nn.Softmax(dim=1)
input = torch.randn(2, 3, 5, 5)
output = m(input)
print(output)
# mask testing 
test = np.ones((3, 3))
test[0, 0:] = 0
test[1, 1:] = 0
test[2, 2:] = 0
print(test)
mask = np.zeros((3, 3, 5, 5))
# h, w
mask[:, :, :2, :] = 1
mask[:, :, 2, :2] = 1
# o, i
mask[0, 0:, :, :] = 0
mask[1, 1:, :, :] = 0
mask[2, 2:, :, :] = 0
print(mask)
'''
#-------------------------------------------------------
'''
test = np.ones((3, 3))
test[0, 0:] = 0
test[1, 1:] = 0
test[2, 2:] = 0
print(test)
mask = np.ones((6, 6, 5, 5))
# h, w
mask[:, :, 2+1:, :] = 0
mask[:, :, 2, 2+1:] = 0
# o, i
mask[:2, 2:, :, :] = 0
mask[:4, 4:, :, :] = 0
# diagonal blocks for type B
mask[0:2, 0:2, :, :] = 0
mask[2:4, 2:4, :, :] = 0
mask[4:, 4:, :, :] = 0
print(mask)
'''

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, batch_size):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.mask_type = mask_type
        self.padding = padding
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.create_mask()

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        return out

    def create_mask(self):
        self.mask = np.ones((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        # weight is shape (out_channels, in_channels, H, W)
        # mask H, W
        self.mask[:, :, self.kernel_size//2+1:, :] = 0
        self.mask[:, :, self.kernel_size//2, self.kernel_size//2+1:] = 0
        # conditional masking
        m = self.out_channels // 3
        n = self.in_channels // 3
        self.mask[:1*m, 1*n:, self.kernel_size//2, self.kernel_size//2] = 0
        self.mask[:2*m, 2*n:, self.kernel_size//2, self.kernel_size//2] = 0
        #print(self.mask[:, :, 0, 0])
        if self.mask_type == 'A':
            # mask H, W
            self.mask[:, :, self.kernel_size // 2, self.kernel_size // 2] = 0
            # conditional masking
            '''
            self.mask[:1*m, :1*n, self.kernel_size//2, self.kernel_size//2] = 0
            self.mask[1*m:2*m, 1*n:2*n, self.kernel_size//2, self.kernel_size//2] = 0
            self.mask[2*m:, 2*n:, self.kernel_size//2, self.kernel_size//2] = 0
            '''

        self.mask = torch.tensor(self.mask, dtype=torch.float32).to(device)

    def create_mask2(self):
        self.mask = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        mask_type = self.mask_type
        self.color_conditioning = True
        k = self.kernel_size
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if self.color_conditioning:
          one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
          if mask_type == 'B':
            self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
            self.mask[one_third_out:2*one_third_out, :2*one_third_in, k // 2, k // 2] = 1
            self.mask[2*one_third_out:, :, k // 2, k // 2] = 1
          else:
            self.mask[one_third_out:2*one_third_out, :one_third_in, k // 2, k // 2] = 1
            self.mask[2*one_third_out:, :2*one_third_in, k // 2, k // 2] = 1
        else:
          if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1
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
    model.append(MaskedConv('B',
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

def train_epoch(dataloader, model, optimizer):
  model.train()
  epoch_loss = []
  for batch in dataloader:
    # batch is shape (N, H, W, C=channels)
    N, H, W, C = batch.shape
    x = torch.tensor(batch, dtype=torch.float32).to(device)
    x = torch.permute(x, (0, 3, 1, 2))
    y = torch.tensor(np.squeeze(batch)).to(device) # shape (N, H, W, C)
    logits = model.forward(x) # shape (N, C_out, H, W)

    # need 12 channels, 3 channels of num_classes=4
    # logits are (N,  C_out, H, W)
    # so we reshape (N, C_out, H, W) -> (N, num_classes, Channels, H, W)
    # and permute target (N, H, W, Channels) -> (N, Channels, H, W)
    N, C_out, H, W = logits.shape
    logits = logits.reshape(N, C_out//C, C, H, W)
    y = torch.permute(y, (0, 3, 1, 2))

    loss = F.cross_entropy(logits, y)

    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(loss)
  return epoch_loss

def test(dataloader, model):
  model.eval()
  epoch_loss = []
  with torch.no_grad():
    for batch in dataloader:
      N, H, W, C = batch.shape
      x = torch.tensor(batch, dtype=torch.float32).to(device)
      x = torch.permute(x, (0, 3, 1, 2))
      y = torch.tensor(np.squeeze(batch)).to(device) # shape (128, 20, 20, 3)

      logits = model(x) # shape (N, out channels, H, W)

      N, C_out, H, W = logits.shape
      logits = logits.reshape(N, C_out//C, C, H, W)

      y = torch.permute(y, (0, 3, 1, 2))

      loss = F.cross_entropy(logits, y)
      epoch_loss.append(loss.item())
    return torch.mean(torch.tensor(epoch_loss))

def sample(model, num_samples, image_size):
  '''
  :param image_size: (C, H, W)
  :return: samples
  '''
  model.eval()
  with torch.no_grad():
    C, H, W = image_size
    N = num_samples
    samples = np.zeros((N, H, W, C))
    for h in range(H):
      for w in range(W):
        for c in range(C):
            x = torch.tensor(samples, dtype=torch.float32).to(device)
            x = torch.permute(x, (0, 3, 1, 2))
            logits = model(x)  # shape (N, out channels, H, W)
            N, C_out, H, W = logits.shape
            logits = logits.reshape(N, C_out // C, C, H, W)  # reshape (N, num_classes, C, H, W)
            logits = torch.permute(logits, (0, 2, 3, 4, 1))  # shape (N, C, H, W, num_classes)
            # sample
            pred_logits = logits[:, c, h, w, :] # shape (100, 2)
            pred_probs = F.softmax(pred_logits, dim=1)
            preds = torch.multinomial(pred_probs, 1, replacement=True).cpu().detach().numpy().squeeze()

            samples[:, h, w, c] = np.array(preds)
      print(h)
  return samples
  #return samples

def sample2(model, num_samples, image_size):
  '''
  :param image_size: (C, H, W)
  :return: samples
  '''
  model.eval()
  with torch.no_grad():
    C, H, W = image_size
    N = num_samples
    samples = torch.zeros((N, H, W, C))
    for h in range(H):
      print(h)
      for w in range(W):
        for c in range(C):
            x = torch.tensor(samples, dtype=torch.float32).to(device)
            x = torch.permute(x, (0, 3, 1, 2))
            logits = model(x)  # shape (N, out channels, H, W)
            N, C_out, H, W = logits.shape
            logits = logits.reshape(N, C_out // C, C, H, W)  # reshape (N, num_classes, C, H, W)
            logits = logits[:, :, c, h, w]
            # model outputs: out.view(batch_size, self.n_channels, self.n_colors,
            #                      *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
            # = (N, C, num_classes, H, W) -> (N, num_classes, C, H, W)
            probs = F.softmax(logits, dim=1)
            samples[:, h, w, c] = torch.multinomial(probs, 1).squeeze(-1)
    samples = samples.cpu().numpy()
    print(samples.shape)
    return samples



def q3_c(train_data, test_data, image_shape, dset_id):
    """
  train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
  test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
  image_shape: (H, W, C), height, width, and # of channels of the image
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
  """

    """ YOUR CODE HERE """
    H, W, C = image_shape
    EPOCHS = 2
    train_losses = []
    test_losses = []
    samples = []

    # image_dim is (in channels, H, W)
    # PixelCNN __init__(self, num_blocks, num_filters, filter_size, batch_size, image_dim, output_channels)
    model = PixelCNN(5, 120, 7, 128, (C, H, W), C*4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    init_loss = test(test_dataloader, model)
    test_losses.append(init_loss)
    print(init_loss)

    for epoch in range(EPOCHS):
      train_losses += train_epoch(train_dataloader, model, optimizer)
      test_losses.append(test(test_dataloader, model))
      print(test_losses[-1])
    if EPOCHS == 0:
      train_losses = [1]
      test_losses = [1]
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    samples = sample(model, num_samples=100, image_size=(C, H, W))
    return train_losses, test_losses, samples
q3bc_save_results(2, 'c', q3_c)
'''
inputs = torch.randn(1, 4, 5, 5) # shape (batch, in channels, input H, input W)
filters = torch.randn(8, 4, 3, 3) # shape (out channels, in channels/groups, kernel H, kernel W)
out = F.conv2d(inputs, filters, padding=1) 
print(out.shape) -> (1, 8, 5, 5) = (batch, out_channels, H*, W*)
'''
