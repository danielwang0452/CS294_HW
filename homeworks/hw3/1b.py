from deepul.hw3_helper import *
import torch
import torch.nn as nn
import numpy as np
device = 'mps'
dtype = torch.float32
#visualize_q1_data('a', 1)
#visualize_q1_data('a', 2)
#visualize_q1_data('b', 1)
#visualize_q1_data('b', 2)
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.q = nn.Sequential(
          nn.Linear(2, self.hidden_size),
          nn.ReLU(),
          nn.Linear(self.hidden_size, self.hidden_size),
          nn.ReLU(),
          nn.Linear(self.hidden_size, 4)
        )
        self.p = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 4)
        )

    def x_z(self, x): # q
        # x is shape (N, 2)
        out = self.q(x)
        mu = out[:, :2]
        logstd = out[:, 2:]
        return mu, logstd

    def z_x(self, z): # p
        out = self.p(z)
        mu = out[:, :2]
        logstd = out[:, 2:]
        return mu, logstd

    def log_prob(self, x):
        z_mu, z_logstd = self.x_z(x) # q
        z_prior = torch.distributions.normal.Normal(torch.tensor(0, dtype=dtype), torch.tensor(1,dtype=dtype))
        z_sample = z_mu + z_logstd.exp() * z_prior.sample(sample_shape=z_mu.shape).to(device)
        x_mu, x_logstd = self.z_x(z_sample) # p

        x_dist = torch.distributions.normal.Normal(x_mu, x_logstd.exp())
        z_dist = torch.distributions.normal.Normal(z_mu, z_logstd.exp())
        # reparameterization trick

        reconstruction_loss = x_dist.log_prob(x).mean()

        #  DKL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        DKL = -0.5 * torch.mean(1 + 2*z_logstd - z_mu.pow(2) - z_logstd.exp().pow(2))
        # loss1 from CS294
        log_px_1 = x_dist.log_prob(x) -z_dist.log_prob(z_sample) + z_prior.log_prob(z_sample)
        # loss2 from CS182 (reconstruction loss - DKL)
        #print(reconstruction_loss.shape)
        #print(DKL.shape)
        log_px_2 = reconstruction_loss + DKL
        #print(DKL.mean())
        return -log_px_2, -reconstruction_loss, DKL

    def sample(self, shape, decoder_noise):
        mean = torch.zeros(shape, dtype=dtype).to(device)
        std = torch.ones(shape, dtype=dtype).to(device)
        z = torch.normal(mean, std)
        mu, sigma = self.z_x(z)
        if decoder_noise == True:
            samples = torch.normal(mu, sigma)
        elif decoder_noise == False:
            samples = torch.normal(mu, torch.zeros_like(mu, dtype=dtype).to(device))
        return samples

def test(model, dataloader):
    model.eval()
    losses = []
    recon_losses = []
    DKLs = []
    with torch.no_grad():
        for batch in dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)

            loss, recon_loss, DKL = model.log_prob(x)
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            DKLs.append(DKL.item())
        print(f'test loss: {loss.item()}')

        return np.array(losses).mean(), np.array(recon_losses).mean(), np.array(DKLs).mean()

def train(train_dataloader, test_dataloader, model, epochs, optimizer):
    losses = []
    recon_losses = []
    DKLs = []
    test_losses = []
    test_recon_losses = []
    test_DKLs = []
    l, r, d = test(model, test_dataloader)
    test_losses.append(l)
    test_recon_losses.append(r)
    test_DKLs.append(d)
    for i in range(epochs):
        for batch in train_dataloader:
            x = torch.tensor(batch, dtype=dtype).to(device)

            loss, recon_loss, DKL = model.log_prob(x)
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            DKLs.append(DKL.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f'loss: {loss.item()}')

        l, r, d = test(model, test_dataloader)
        test_losses.append(l)
        test_recon_losses.append(r)
        test_DKLs.append(d)

    return np.stack((np.array(losses), np.array(recon_losses), np.array(DKLs)), axis=1), \
           np.stack((np.array(test_losses), np.array(test_recon_losses), np.array(test_DKLs)), axis=1)

def q1(train_data, test_data, part, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """

    """ YOUR CODE HERE """
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    model = VAE()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 10
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    train_results, test_results = train(train_dataloader, test_dataloader, model, EPOCHS, optimizer)
    samples_1 = model.sample((1000, 2), decoder_noise=True)
    samples_2 = model.sample((1000, 2), decoder_noise=False)
    print(train_results.shape)
    return train_results, test_results, samples_1.cpu().detach().numpy(), samples_2.cpu().detach().numpy()

q2_save_results('a', 1, q2_a)