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

class MLP(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_size, hidden_size)])
        self.net.append(nn.ReLU())
        for n in range(n_hidden_layers):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_size, out_size))

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out

class MixtureFlow(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.loc = nn.Parameter(torch.randn(n_components), requires_grad=True)
        #torch.nn.init.normal_(self.loc)
        self.scale = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        #torch.nn.init.normal_(self.scale)
        self.weight = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        #torch.nn.init.normal_(self.weight)

        self.base_dist = Uniform(0.0, 1.0)
        self.mix_dist = Normal

    def forward(self, x):
        scale_weight = F.softmax(self.weight, dim=0).unsqueeze(0).repeat(x.shape[0], 1)

        z = (self.mix_dist(self.loc, self.scale.exp()).cdf(
            x.repeat(1, self.n_components))*scale_weight).sum(dim=1)
         # log det = log dz/dx = log pdf(x)
        log_det = (self.mix_dist(self.loc, self.scale.exp()).log_prob(
            x.repeat(1, self.n_components)).exp()*scale_weight).sum(dim=1).log()
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base_dist.log_prob(z).to(device) + log_det

    def loss(self, x):
        return -self.log_prob(x).mean()

class AutoregressiveFlow(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.z1Flow = MixtureFlow(n_components).to(device)
        self.MLP = MLP(in_size=1, n_hidden_layers=3, hidden_size=64, out_size=n_components*3).to(device)
        self.base_dist = torch.distributions.uniform.Uniform(0.0, 1.0)
        self.mix_dist = torch.distributions.normal.Normal

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        # z1
        z1, z1_det = self.z1Flow.forward(x1) # x is already squeezed

        # z2
        loc, scale, weights = torch.chunk(self.MLP(x1), chunks=3, dim=1)
        scale_weight = F.softmax(weights, dim=1)
        z2 = (self.mix_dist(loc, scale.exp()).cdf(
            x2.repeat(1, self.n_components)) * scale_weight).sum(dim=1)
        # log det = log dz/dx = log pdf(x)

        z2_det = (self.mix_dist(loc, scale.exp()).log_prob(
            x2.repeat(1, self.n_components)).exp() * scale_weight).sum(dim=1).log()
        return torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1), \
            torch.cat([z1_det.unsqueeze(1), z2_det.unsqueeze(1)], dim=1)

    def log_prob(self, x):
        z, z_det = self.forward(x.squeeze()) # shape: [batch_size, dim]
        return (self.base_dist.log_prob(z*0.999999 + 1e-8) + z_det).mean(dim=1)

    def loss(self, x):
        return -self.log_prob(x).mean()

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
        #print(loss)
        return loss

def q1_a(train_data, test_data, dset_id):
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
    EPOCHS = 100
    train_losses = []
    test_losses = []
    # create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # model
    model = AutoregressiveFlow(n_components=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
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

q1_save_results(1, 'a', q1_a)