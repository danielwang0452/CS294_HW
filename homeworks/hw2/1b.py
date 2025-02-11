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

device = 'cpu'
dtype = torch.float32

def train(model, train_dataloader, test_dataloader, epochs, optimizer):

    train_losses = []
    test_losses = []
    test_losses.append(test(model, test_dataloader))

    for epoch in range(epochs):
        model.train()
        #print(epoch)
        for batch in train_dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device).float().contiguous()
            loss = model.loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        #print(f'train loss: {loss}')

        test_losses.append(test(model, test_dataloader))
    return np.array(train_losses), np.array(test_losses)

def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        epoch_losses = []
        for batch in dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device).float().contiguous()
            loss = model.loss(x)
            epoch_losses.append(loss.item())
        loss = np.array(epoch_losses).mean()
        print(f'test loss: {loss}')
        return loss

class MLP(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_size, hidden_size)])
        for n in range(n_hidden_layers-1):
            self.net.append(nn.ReLU())
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_size, out_size))

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out

'''
class AffineTransform(nn.Module):
    def __init__(self, type, n_hidden=2, hidden_size=256):
        super().__init__()
        self.mask = self.build_mask(type=type)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mlp = MLP(in_size=2, n_hidden_layers=n_hidden, hidden_size=hidden_size, out_size=2)

    def build_mask(self, type):
        # if type == "left", left half is a one
        # if type == right", right half is a one
        assert type in {"left", "right"}
        if type == "left":
            mask = ptu.FloatTensor([1.0, 0.0])
        elif type == "right":
            mask = ptu.FloatTensor([0.0, 1.0])
        else:
            raise NotImplementedError
        return mask

    def forward(self, x, reverse=False):
        # returns transform(x), log_det
        batch_size = x.shape[0]
        mask = self.mask.repeat(batch_size, 1)
        x_ = x * mask

        log_s, t = self.mlp(x_).split(1, dim=1) #todo: fix
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        #print(x[0])
        #print(log_s[0])
        return x, log_s

class RealNVP2(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
        self.net = nn.ModuleList([])
        for n in range(n_layers):
            if (-1)**n == 1:
                type = 'left'
            elif (-1)**n == -1:
                type = 'right'
            self.net.append(AffineTransform(type=type,
                                          n_hidden=2,
                                          hidden_size=64))

    def forward(self, x):
        z, log_det = x, torch.zeros_like(x)
        for layer in self.net:
            # flip first layer, so last layer is not flipped if there are even layers
            z, delta_log_det = layer.forward(z)
            log_det += delta_log_det
        return z, log_det # shape (batch size, dim), (batch size)

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return torch.sum(log_det, dim=1) + torch.sum(self.prior.log_prob(z), dim=1)

    def loss(self, x):
        return -self.log_prob(x).mean() # avg across batch
'''
class CouplingLayer(nn.Module):
    def __init__(self, nn_in_size, nn_hidden_size, nn_num_layers):
        super().__init__()
        self.g = MLP(in_size=nn_in_size,
                     n_hidden_layers=nn_num_layers,
                     hidden_size=nn_hidden_size,
                     out_size=nn_in_size)
        self.h = MLP(in_size=nn_in_size,
                     n_hidden_layers=nn_num_layers,
                     hidden_size=nn_hidden_size,
                     out_size=nn_in_size)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        #torch.nn.init.normal_(self.scale)
        #torch.nn.init.normal_(self.scale_shift)

    def forward(self, x1, x2):
        y1 = x1
        scale = self.h(x1)
        log_scale = self.scale*torch.tanh(scale) + self.scale_shift
        y2 = x2 * log_scale.exp() + self.g(x1)

        log_det = log_scale
        #print(log_det[0])
        return torch.cat([y1, y2], dim=1), log_det

class RealNVP(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
        self.net = nn.ModuleList([])
        for n in range(n_layers):
            self.net.append(CouplingLayer(nn_in_size=1,
                                          nn_hidden_size=64,
                                          nn_num_layers=2))

    def forward(self, x):
        y = x
        log_det = torch.zeros(x.shape[0], dtype=dtype).to(device)
        for l, layer in enumerate(self.net):
            y1, y2 = y.chunk(2, dim=1)
            # flip first layer, so last layer is not flipped if there are even layers
            if (-1)**l > 0: # need to alternate/flip y1, y2 after every block
                # switch
                y, y_det = layer(y2, y1)
            else:
                # no switch
                y, y_det = layer(y1, y2)

            log_det += y_det.squeeze()
        z = y
        return z, log_det # shape (batch size, dim), (batch size)

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return log_det + self.prior.log_prob(z).sum(dim=1) # sum across each dimension

    def loss(self, x):
        return -self.log_prob(x).mean() # avg across batch

def q1_b(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets, or
               for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity).
        Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents
        mapping the train set data points through our flow to the latent space.
    """
    """ YOUR CODE HERE """
    EPOCHS = 350
    train_losses = []
    test_losses = []
    # create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # model
    model = RealNVP(n_layers=6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # train
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, EPOCHS, optimizer)
    # heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)

    elif dset_id == 2:  # two moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = ptu.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))

    densities = np.exp(ptu.get_numpy(model.log_prob(mesh_xs.to(device)).cpu()))

    # latents
    z, _ = model.forward(ptu.FloatTensor(train_data))
    latents = ptu.get_numpy(z)
    return train_losses, test_losses, densities, np.array(latents)

q1_save_results(1, 'b', q1_b)