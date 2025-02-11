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

class MaskedConv1D(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_type = mask_type

        self.create_mask()

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        # (shape batch_size, out_channels, W)
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
        # zero out all other rows
        self.mask[:, :, :self.kernel_size, :] = 0
        self.mask[:, :, self.kernel_size+1:, :] = 0

        self.mask = torch.tensor(self.mask, dtype=torch.float32).to(device)

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, cond_in_channels):
        super().__init__(in_channels, out_channels, kernel_size, padding) # calls constructor of nn.Conv2D with *args parameters
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_type = mask_type
        self.cond_embedding = nn.Conv2d(in_channels=cond_in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        padding=1)

        self.create_mask()

    def forward(self, x, cond):
        if cond is None:
            out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        else:
            out = F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding) + self.cond_embedding(cond)
        # (shape batch_size, out_channels, h, w)
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

        self.mask = torch.tensor(self.mask, dtype=torch.float32).to(device)

class StackLayerNorm(nn.Module):
  def __init__(self, normalized_shape):
    super().__init__()
    self.h_layer_norm = LayerNorm(normalized_shape)
    self.v_layer_norm = LayerNorm(normalized_shape)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)
    vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
    return torch.cat((vx, hx), dim=1)

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape[1])
    # layernorm only applied across the 64 channels
    def forward(self, x):
        # x is shape (N, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        return out.permute(0, 3, 1, 2).contiguous()

class GatedLayer(nn.Module):
    def __init__(self, kernel_size, p, mask_type, cond_in_channels):
        super().__init__()
        # hyperparams
        self.kernel_size = kernel_size
        self.p = p
        self.mask_type = mask_type
        # conv layers
        self.n_conv = MaskedConv(mask_type='B',
                                 in_channels=self.p,
                                 out_channels=self.p*2,
                                 kernel_size=self.kernel_size,
                                 padding=self.kernel_size//2,
                                 cond_in_channels=cond_in_channels)
        self.cross_conv = MaskedConv(mask_type='B',
                                     in_channels=self.p*2,
                                     out_channels=self.p*2,
                                     kernel_size=1,
                                     padding=0,
                                     cond_in_channels=cond_in_channels)
        self.one_conv = MaskedConv(mask_type='B',
                                   in_channels=self.p,
                                   out_channels=self.p,
                                   kernel_size=1,
                                   padding=0,
                                   cond_in_channels=cond_in_channels)
        self.conv1D = MaskedConv1D(mask_type=self.mask_type,
                                   in_channels=self.p,
                                   out_channels=self.p*2,
                                   kernel_size=self.kernel_size,
                                   padding=self.kernel_size//2,
                                   )

    def forward(self, x, cond):
        # x is shape (N, C_in=2p, H, W)
        # out is shape (shape N, 2p, H, W)
        N, C_in, H, W = x.shape
        s = nn.Sigmoid()
        t = nn.Tanh()

        vx, hx = x.chunk(2, dim=1)

        A = self.n_conv(vx, cond)
        B = self.cross_conv(A, cond) + self.conv1D(hx)

        A1, A2 = A.chunk(2, dim=1)
        B1, B2 = B.chunk(2, dim=1)
        right_out = (self.one_conv(t(B1) * s(B2), cond)) + hx
        left_out = t(A1) * s(A2)

        return torch.concatenate((right_out, left_out), dim=1) # (shape N, 2p, H, W)

class GatedPixelCNN(nn.Module):
    def __init__(self, num_layers, p, kernel_size, in_channels, out_channels, batch_size, H, W, cond_in_channels):
        super().__init__()
        # params
        self.num_layers = num_layers
        self.p = p
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.H = H
        self.W = W
        # model layers
        self.in_conv = MaskedConv(mask_type='A',
                                  in_channels=in_channels,
                                  out_channels=p,
                                  kernel_size=7,
                                  padding=3,
                                  cond_in_channels=cond_in_channels)

        self.out_conv = MaskedConv(mask_type='B',
                                  in_channels=p,
                                  out_channels=out_channels,
                                  kernel_size=7,
                                  padding=3,
                                  cond_in_channels=cond_in_channels)
        self.net = nn.ModuleList()
        for n in range(num_layers-2):
            #self.net.append(nn.ReLU())
            self.net.append(GatedLayer(self.kernel_size, self.p, mask_type='B',cond_in_channels=cond_in_channels))
            self.net.append(StackLayerNorm((self.batch_size, self.p, self.H, self.W)))

    def forward(self, x, cond):
        out = self.in_conv(x, cond)
        out = torch.cat((out, out), dim=1)
        for layer in self.net:
            if isinstance(layer, GatedLayer):
                out = layer(out, cond)
            else:
                out = layer(out)
        out = self.out_conv(out.chunk(2, dim=1)[1], cond)
        return out

def train_epoch(dataloader, model, optimizer):
  model.train()
  epoch_loss = []
  for batch in dataloader:
    # preprocess binary img
    binary_image = to_binary(batch.permute((0, 3, 1, 2)))
    x_binary = torch.tensor(binary_image, dtype=torch.float32).to(device)
    y_binary = torch.tensor(np.squeeze(binary_image), dtype=torch.int).to(device)
    # batch is shape (N, H, W, C=channels)
    N, H, W, C = batch.shape
    x = torch.tensor(batch, dtype=torch.float32).to(device)
    x = torch.permute(x, (0, 3, 1, 2))
    y = torch.tensor(np.squeeze(batch)).to(device)  # shape (N, H, W, C)
    binary_logits, colour_logits = model.forward(x_binary, x)  # shape (N, C_out, H, W)

    # need 12 channels, 3 channels of num_classes=4
    # logits are (N,  C_out, H, W)
    # so we reshape (N, C_out, H, W) -> (N, num_classes, Channels, H, W)
    # and permute target (N, H, W, Channels) -> (N, Channels, H, W)
    N, C_out, H, W = colour_logits.shape
    colour_logits = colour_logits.reshape(N, C_out // C, C, H, W)
    y = torch.permute(y, (0, 3, 1, 2))
    binary_loss = F.cross_entropy(binary_logits, y_binary)
    colour_loss = F.cross_entropy(colour_logits, y)
    loss = binary_loss + colour_loss

    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

  print(f'binary: {binary_loss} colour: {colour_loss} loss: {loss}')
  return epoch_loss

def test(dataloader, model):
  model.eval()
  epoch_loss = []
  with torch.no_grad():
    for batch in dataloader:
        # preprocess binary img
        binary_image = to_binary(batch.permute((0, 3, 1, 2)))
        x_binary = torch.tensor(binary_image, dtype=torch.float32).to(device)
        y_binary = torch.tensor(np.squeeze(binary_image), dtype=torch.int).to(device)
        # batch is shape (N, H, W, C=channels)
        N, H, W, C = batch.shape
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        x = torch.permute(x, (0, 3, 1, 2))
        y = torch.tensor(np.squeeze(batch)).to(device)  # shape (N, H, W, C)
        binary_logits, colour_logits = model.forward(x_binary, x)  # shape (N, C_out, H, W)

        # need 12 channels, 3 channels of num_classes=4
        # logits are (N,  C_out, H, W)
        # so we reshape (N, C_out, H, W) -> (N, num_classes, Channels, H, W)
        # and permute target (N, H, W, Channels) -> (N, Channels, H, W)
        N, C_out, H, W = colour_logits.shape
        colour_logits = colour_logits.reshape(N, C_out // C, C, H, W)
        y = torch.permute(y, (0, 3, 1, 2))
        binary_loss = F.cross_entropy(binary_logits, y_binary)
        colour_loss = F.cross_entropy(colour_logits, y)
        loss = binary_loss + colour_loss

        epoch_loss.append(loss.item())
    return torch.mean(torch.tensor(epoch_loss))

def sample(model, num_samples, image_size, cond):
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

            logits = model(x, cond)  # shape (N, out channels, H, W)
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

def to_binary(images):
  images = images.float()
  # To grayscale
  images = 0.3 * images[:, [0], :, :] + 0.59 * images[:, [1], :, :] + 0.11 * images[:, [2], :, :]
  # Scale to [0, 1]
  images /= 3
  # Binarize
  images = (images > 0.5).float()
  return images

class GrayscalePixelCNN(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W

        self.binary_model = GatedPixelCNN(num_layers=4, p=64 , kernel_size=7, in_channels=1, out_channels=2,
                          batch_size=128, H=H, W=W, cond_in_channels=1)

        self.colour_model = GatedPixelCNN(num_layers=3, p=64 , kernel_size=7, in_channels=3, out_channels=12,
                          batch_size=128, H=H, W=W, cond_in_channels=1)

    def forward(self, binary_image, colour_image):
        binary = self.binary_model(binary_image, None)
        colour = self.colour_model(colour_image, binary_image)
        return binary, colour

    def sample(self, num_samples):
        binary_samples = sample(self.binary_model, num_samples, image_size=(1, self.H, self.W), cond=None)
        colour_samples = sample(self.colour_model, num_samples, image_size=(3, self.H, self.W),
                                cond=torch.tensor(binary_samples, dtype=torch.float32).permute(0, 3, 1, 2).to(device))
        return binary_samples, colour_samples

def q4_b(train_data, test_data, image_shape):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (50, H, W, 1) of generated binary images in {0, 1}
    - a numpy array of size (50, H, W, C) of conditonally generated color images in {0, 1, 2, 3}
    """
    # You will need to generate the binary image dataset from train_data and test_data

    """ YOUR CODE HERE """
    H, W, C = image_shape
    EPOCHS = 2
    train_losses = []
    test_losses = []
    samples = []

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # image_dim is (in channels, H, W)
    # PixelCNN __init__(self, num_blocks, num_filters, filter_size, batch_size, image_dim, output_channels)
    model = GrayscalePixelCNN(H, W)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

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

    binary_samples, colour_samples = model.sample(50)

    return train_losses, test_losses, binary_samples, colour_samples

q4b_save_results(q4_b)


