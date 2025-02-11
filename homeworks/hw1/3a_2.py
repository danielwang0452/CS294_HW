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

#-------------------------------------------------------
#visualize_q2a_data(dset_type=1)
#visualize_q2a_data(dset_type=2)

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, batch_size):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.mask_type = mask_type
        self.padding = padding
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.create_mask()
    def forward(self, x):
        out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding) # need to implement mask
        return out

    def create_mask(self):
        # weight is shape (out_channels, in_channels, H, W)
        self.mask = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.mask[:, :, :self.kernel_size//2, :] = 1
        self.mask[:, :, self.kernel_size//2, :self.kernel_size//2] = 1
        if self.mask_type == 'B':
            self.mask[:, :, self.kernel_size // 2, self.kernel_size // 2] = 1
        self.mask = torch.tensor(self.mask, dtype=torch.float32).to(device)

class PixelCNN(nn.Module):
  def __init__(self, num_hidden_layers, num_one_layers, num_filters, filter_size, batch_size, image_dim, output_channels):
    super().__init__()
    self.num_hidden_layers = num_hidden_layers
    self.num_filters = num_filters # num filters in each layer = output channels
    self.filter_size = filter_size # should be an odd integer
    self.batch_size = batch_size
    self.in_channels, self.iH, self.iW = image_dim
    self.output_channels = output_channels # this is for output layer
    self.num_one_layers = num_one_layers
    # build network
    # input layer
    model = nn.ModuleList([MaskedConv('A',
                                      in_channels=self.in_channels,
                                      out_channels=self.num_filters,
                                      kernel_size=self.filter_size,
                                      padding=self.filter_size//2,
                                      batch_size=self.batch_size)],
                                      )
    for i in range(self.num_hidden_layers):
        model.append(MaskedConv(mask_type='B',
                                in_channels=self.num_filters,
                                out_channels=self.num_filters,
                                kernel_size=self.filter_size,
                                padding=self.filter_size//2,
                                batch_size=self.batch_size))
    # output layer
    model.append(MaskedConv(mask_type='B',
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
        if l < len(self.net) - 1: # apply ReLU before last layer
            out = a(layer(out))
        else:
            out = layer(out) # no ReLU on last layer
    return out

def train_epoch(dataloader, model, optimizer):
    model.train()
    epoch_loss = []
    for batch in dataloader:
        # batch is shape (N, H, W, C=1)
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        x = torch.permute(x, (0, 3, 1, 2))
        y = torch.tensor(np.squeeze(batch)).to(device) # (N, H, W)
        logits = model.forward(x)  # shape (N, out channels, H, W)
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
      x = torch.tensor(batch, dtype=torch.float32).to(device)
      x = torch.permute(x, (0, 3, 1, 2))
      y = torch.tensor(np.squeeze(batch)).to(device)
      logits = model(x)
      loss = F.cross_entropy(logits, y)
      epoch_loss.append(loss.item())
    return torch.mean(torch.tensor(epoch_loss))

def sample(model, num_samples, image_size):
  '''
  :param image_size: (C=1, H, W)
  :return: samples
  '''
  model.eval()
  with torch.no_grad():
    C, H, W = image_size
    N = num_samples
    samples = np.zeros((N, H, W, C))
    for h in range(H):
      for w in range(W):
        x = torch.tensor(samples, dtype=torch.float32).to(device)
        x = torch.permute(x, (0, 3, 1, 2))
        logits = model(x) # shape (N, out channels, H, W)
        logits = torch.permute(logits, (0, 2, 3, 1)) # shape (N, H, W, C)
        pred_logits = logits[:, h, w, :] # shape (100, 2)
        pred_probs = F.softmax(pred_logits, dim=1)

        preds = torch.multinomial(pred_probs, 1, replacement=True).cpu().detach().numpy()
        '''
        preds = []
        for n in pred_probs:
          pred = torch.multinomial(n, num_samples=1, replacement=True)
          preds.append([pred.item()])
        '''
        samples[:, h, w, :] = np.array(preds)
      print(h)
  return samples

def q3_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """
    """ YOUR CODE HERE """
    H, W = image_shape
    EPOCHS = 10
    train_losses = []
    test_losses = []
    samples = []
    # model params (num_hidden_layers, num_filters, filter_size, batch_size, image_dim, out_channels)
    # image_dim is (in channels, H, W)
    model = PixelCNN(5, 2, 64, 7, 128, (1, H, W), 2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)  # 32272324

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
    samples = sample(model, num_samples=100, image_size=(1, H, W))
    return train_losses, test_losses, samples
q3a_save_results(1, q3_a)
'''
inputs = torch.randn(1, 4, 5, 5) # shape (batch, in channels, input H, input W)
filters = torch.randn(8, 4, 3, 3) # shape (out channels, in channels/groups, kernel H, kernel W)
out = F.conv2d(inputs, filters, padding=1) 
print(out.shape) -> (1, 8, 5, 5) = (batch, out_channels, H*, W*)
'''
