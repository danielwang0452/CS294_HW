from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'mps'

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, batch_size, num_classes):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.mask_type = mask_type
        self.padding = padding
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.create_mask()
        self.cond_op = nn.Linear(num_classes, out_channels, bias=True)

    def forward(self, x, y):
        condition = self.cond_op(y) # shape (batch_size, out_channels)
        out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        # (shape batch_size, out_channels, h, w)
        return out + condition.unsqueeze(-1).unsqueeze(-1)

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
    def __init__(self, mid_kernel_size, in_channels, batch_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.mid_kernel_size = mid_kernel_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.block = nn.ModuleList([MaskedConv('B',
                                      in_channels=self.in_channels,
                                      out_channels=self.in_channels//2,
                                      kernel_size=1,
                                      padding=0,
                                      batch_size=self.batch_size,
                                      num_classes=self.num_classes),
                                    nn.ReLU(),
                                    MaskedConv('B',
                                      in_channels=self.in_channels//2,
                                      out_channels=self.in_channels // 2,
                                      kernel_size=self.mid_kernel_size,
                                      padding=self.mid_kernel_size//2,
                                      batch_size=self.batch_size,
                                      num_classes=self.num_classes),
                                    nn.ReLU(),
                                    MaskedConv('B',
                                      in_channels=self.in_channels // 2,
                                      out_channels=self.in_channels,
                                      kernel_size=1,
                                      padding=0,
                                      batch_size=self.batch_size,
                                      num_classes=self.num_classes),
                                    nn.ReLU()
                                    ],
                                      )

    def forward(self, x, y):
        # x is shape (N, C, H, W)
        out = x
        for layer in self.block:
            if isinstance(layer, MaskedConv):
                out = layer(out, y)
            else:
                out = layer(out)
        return out + x

class PixelCNN(nn.Module):
  def __init__(self, num_blocks, num_filters, filter_size, batch_size, image_dim, output_channels, num_classes):
    super().__init__()
    self.num_blocks = num_blocks
    self.num_filters = num_filters # num filters in each layer = output channels
    self.filter_size = filter_size # should be an odd integer
    self.batch_size = batch_size
    self.in_channels, self.iH, self.iW = image_dim
    self.num_classes = num_classes
    self.output_channels = output_channels # this is for output layer
    # build network
    # input layer
    model = nn.ModuleList([MaskedConv('A',
                                      in_channels=self.in_channels,
                                      out_channels=self.num_filters,
                                      kernel_size=self.filter_size,
                                      padding=self.filter_size//2,
                                      batch_size=self.batch_size,
                                      num_classes=self.num_classes),
                           nn.ReLU()])
    # resblocks
    for i in range(self.num_blocks):
        # resblock __init__(self, mid_kernel_size, in_channels, batch_size):
        model.append(ResBlock(7, self.num_filters, self.batch_size, num_classes=self.num_classes))
        model.append(LayerNorm((self.batch_size, self.num_filters, self.iH, self.iW)))
    # output layer
    model.append(MaskedConv('B',
               in_channels=self.num_filters,
               out_channels=self.output_channels,
               kernel_size=self.filter_size,
               padding=self.filter_size // 2,
               batch_size=self.batch_size,
               num_classes=self.num_classes))
    self.net = model

  def forward(self, x, y):
    '''
    :param x: shape (batch size, in channels, iH, iW)
    :param y: shape (batch size, num_classes)
    :return: shape (batch size, output channels, iH, iW)
    '''
    out = x.to(device)
    a = nn.ReLU()
    for l, layer in enumerate(self.net):
        if isinstance(layer, MaskedConv) or isinstance(layer, ResBlock):
            out = layer(out, y)
        else:
            out = layer(out)
    return out

def train_epoch(dataloader, model, optimizer):
  model.train()
  epoch_loss = []
  for batch in dataloader:
      # batch is [sample, label]
      # sample is (N, C, H, W) label is (N, n_classes)
      x = torch.tensor(batch[0], dtype=torch.float32).to(device)
      condition = torch.tensor(batch[1], dtype=torch.float32).to(device)
      N, C, H, W = x.shape
      _, num_classes = condition.shape
      logits = model(x, condition)  # shape (N, out channels, H, W)
      N, C_out, H, W = logits.shape
      logits = logits.reshape(N, C_out // C, C, H, W)
      loss = F.cross_entropy(logits, batch[0].to(device))
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
      # batch is [sample, label]
      # sample is (N, C, H, W) label is (N, n_classes)
      x = torch.tensor(batch[0], dtype=torch.float32).to(device)
      condition = torch.tensor(batch[1], dtype=torch.float32).to(device)
      N, C, H, W = x.shape
      _, num_classes = condition.shape
      logits = model(x, condition) # shape (N, out channels, H, W)
      N, C_out, H, W = logits.shape
      logits = logits.reshape(N, C_out//C, C, H, W)
      loss = F.cross_entropy(logits, batch[0].to(device))
      epoch_loss.append(loss.item())
    return torch.mean(torch.tensor(epoch_loss))

def sample(model, num_samples, image_size, num_classes):
  '''
  :param image_size: (C=1, H, W)
  :return: samples
  '''
  # create labels
  y = np.zeros((num_samples, num_classes))
  for n in range(num_classes):
    n_per_class = num_samples//num_classes
    y[n*n_per_class:(n+1)*n_per_class, n] = 1
  y = torch.tensor(y, dtype=torch.float32).to(device)

  model.eval()
  with torch.no_grad():
    C, H, W = image_size
    N = num_samples
    samples = np.zeros((N, H, W, C))
    for h in range(H):
      for w in range(W):
        x = torch.tensor(samples, dtype=torch.float32).to(device)
        x = torch.permute(x, (0, 3, 1, 2))
        logits = model(x, y) # shape (N, out channels, H, W)
        logits = torch.permute(logits, (0, 2, 3, 1)) # shape (N, H, W, C)
        pred_logits = logits[:, h, w, :] # shape (100, 2)
        pred_probs = F.softmax(pred_logits, dim=1)
        preds = torch.multinomial(pred_probs, 1, replacement=True).cpu().detach().numpy()
        samples[:, h, w, :] = np.array(preds)
      print(h)
  return samples

class SimpleDataset(torch.utils.data.Dataset):
  def __init__(self, x, y):
    super().__init__()
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    return self.x[index], self.y[index]

def preprocess_dset(data, labels, n_classes):
  data = np.transpose(data, (0, 3, 1, 2)) # reshape to (N, C, H, W) as required by Conv2D
  labels_oh = np.zeros((len(labels), n_classes))
  labels_oh[np.arange(len(labels)), labels] = 1
  labels_oh = labels_oh.astype('float32')
  return data, labels_oh

def invert_dataset(dataset):
    N, H, W, _ = dataset.shape
    new_dataset = np.ones_like(dataset)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                if dataset[n, h, w] == [1]:
                    new_dataset[n, h, w] = [0]
    return new_dataset

def q3_d(train_data, train_labels, test_data, test_labels, image_shape, n_classes, dset_id):
    """
    train_data: A (n_train, H, W, 1) numpy array of binary images with values in {0, 1}
    train_labels: A (n_train,) numpy array of class labels
    test_data: A (n_test, H, W, 1) numpy array of binary images with values in {0, 1}
    test_labels: A (n_test,) numpy array of class labels
    image_shape: (H, W), height and width
    n_classes: number of classes (4 or 10)
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, 1) of samples with values in {0, 1}
      where an even number of images of each class are sampled with 100 total
    """
    """ YOUR CODE HERE """
    H, W = image_shape
    C = 1
    EPOCHS = 5
    train_losses = []
    test_losses = []
    samples = []

    #train_data = invert_dataset(train_data)
    #test_data = invert_dataset(test_data)

    train_data, train_labels_oh = preprocess_dset(train_data, train_labels, n_classes)
    test_data, test_labels_oh = preprocess_dset(test_data, test_labels, n_classes)

    train_loader = data.DataLoader(SimpleDataset(train_data, train_labels_oh),
                                   batch_size=128, shuffle=True)
    test_loader = data.DataLoader(SimpleDataset(test_data, test_labels_oh), batch_size=128)
    # image_dim is (in channels, H, W)
    # PixelCNN __init__(self, num_blocks, num_filters, filter_size, batch_size, image_dim, output_channels)
    model = PixelCNN(5, 128, 7, 128, (C, H, W), 2, num_classes=n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    init_loss = test(test_loader, model)
    test_losses.append(init_loss)
    print(init_loss)

    for epoch in range(EPOCHS):
        train_losses += train_epoch(train_loader, model, optimizer)
        test_losses.append(test(test_loader, model))
        print(test_losses[-1])
    if EPOCHS == 0:
        train_losses = [1]
        test_losses = [1]
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    samples = sample(model, num_samples=100, image_size=(C, H, W), num_classes=n_classes)
    return train_losses, test_losses, samples

q3d_save_results(2, q3_d)

'''
inputs = torch.randn(1, 4, 5, 5) # shape (batch, in channels, input H, input W)
filters = torch.randn(8, 4, 3, 3) # shape (out channels, in channels/groups, kernel H, kernel W)
out = F.conv2d(inputs, filters, padding=1) 
print(out.shape) -> (1, 8, 5, 5) = (batch, out_channels, H*, W*)
'''


