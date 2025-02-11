from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cpu'

#-------------------------------------------------------
#visualize_q2a_data(dset_type=1)
#visualize_q2a_data(dset_type=2)

class MADE(nn.Module):
  def __init__(self, data_dim, num_hidden_layers):
    super().__init__()
    self.data_dim = data_dim
    self.num_hidden_layers = num_hidden_layers
    self.mask1 = create_mask(1, data_dim)
    self.mask2 = create_mask(2, data_dim)
    # initialise network params
    self.weights = {}
    self.biases = {}
    for i in range(self.num_hidden_layers):
      #self.weights[f'weights_{i}'] = nn.Parameter(torch.rand((data_dim*2, data_dim*2)), requires_grad=True)
      #self.biases[f'biases_{i}'] = nn.Parameter(torch.rand((data_dim*2)), requires_grad=True)
      setattr(self, f'weights_{i}', nn.Parameter(torch.rand((data_dim * 2, data_dim * 2)), requires_grad=True))
      setattr(self, f'biases_{i}', nn.Parameter(torch.rand((data_dim * 2)), requires_grad=True))

  def forward(self, x):
    m = nn.ReLU
    j = 0
    out = m(F.linear(x, getattr(self, f'weights_{0}')*self.mask1, getattr(self, f'biases_{0}')))
    for n in range(len(self.weights)):
      out = m(F.linear(x, getattr(self, f'weights_{n}')*self.mask2, getattr(self, f'biases_{n}')))
      j = n+1
    out = F.linear(x, getattr(self, f'weights_{j}'), getattr(self, f'biases_{j}'))
    return out

def train_epoch(dataloader, model, d, optimizer):
  model.train()
  epoch_loss = []
  for batch in dataloader:
    x = torch.tensor(batch[0], dtype=torch.float32).to(device)
    y = batch[1]
    logits = model.forward(x) # shape (N, 2D)
    # split logits into 2
    loss1 = F.cross_entropy(logits[:, :d], y[:, 0])
    loss2 = F.cross_entropy(logits[:, d:2*d], y[:, 1])
    loss = loss1 + loss2
    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(loss)
  return epoch_loss

def test(dataloader, model, d):
  model.eval()
  epoch_loss = []
  with torch.no_grad():
    for batch in dataloader:
      x = torch.tensor(batch[0], dtype=torch.float32).to(device)
      y = batch[1]
      logits = model(x)  # shape (N, 2D)
      # split logits into 2
      loss1 = F.cross_entropy(logits[:, :d], y[:, 0])
      loss2 = F.cross_entropy(logits[:, d:2 * d], y[:, 1])
      loss = loss1 + loss2
      epoch_loss.append(loss.item())
    return torch.mean(torch.tensor(epoch_loss))

def create_mask(type, d):
  '''
  :param type: 1 or 2
  type 1: strictly lower triangular mask (0 blocks on diagonal)
  type 2: lower triangular (1 blocks on diagonal)
  :return: block matrix mask
  '''
  zero_block = np.zeros((d, d))
  one_block = np.ones((d, d))
  if type == 1:
    top_left = zero_block
    top_right = zero_block
    bottom_left = one_block
    bottom_right = zero_block
  if type == 2:
    top_left = one_block
    top_right = zero_block
    bottom_left = one_block
    bottom_right = one_block
  arr1 = np.concatenate((top_left, top_right), axis=1)
  arr2 = np.concatenate((bottom_left, bottom_right), axis=1)
  mask = np.concatenate((arr1, arr2), axis=0)
  return torch.tensor(mask, dtype=torch.long, requires_grad=False).to(device)

def one_hot_encode(x, d):
  '''
  :param x: e.g [19 23]
  :return:  one hot encoded [ [], [] ]
  '''
  new = []
  for i in x:
    new1 = np.zeros(d)
    new1[i] = 1
    new.append(new1)
  return np.array(new)

def preprocess_dataset(dataset, d):
  '''
  :param dataset: [ ... [2, 3], [5, 2] ... ]
  :return: dataset of [...
                      [[concatenated one hot encoded x0, x1], [labels = x0, x1]]
                      ...]
  '''
  new_dataset = []
  for sample in dataset:
    encoded_sample = one_hot_encode(sample, d)
    encoded_sample = np.concatenate((encoded_sample[0], encoded_sample[1]), axis=0)
    labels = sample
    new_data_point = [encoded_sample, labels]
    new_dataset.append(new_data_point)
  return new_dataset

def get_distribution(model, d):
  distribution = np.zeros((d, d))
  model.eval()
  data = np.mgrid[0:d, 0:d].reshape(2, d ** 2).T
  with torch.no_grad():
    for x in data:
      x_encoded = one_hot_encode(x, d)
      x_encoded = np.concatenate((x_encoded[0], x_encoded[1]), axis=0)
      logits = model(torch.tensor((x_encoded), dtype=torch.float32).to(device)) # shape (N, 2D)
      d1 = F.softmax(logits[:d])
      d2 = F.softmax(logits[d:2*d])
      p1 = d1[x[0]]
      p2 = d2[x[1]]
      distribution[x[0], x[1]] = p1*p2
      distribution = torch.tensor(distribution)
  flatten = F.softmax(distribution.view(-1))
  distribution = flatten.reshape(distribution.shape)
  return distribution.numpy()

def q2_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1} = (8000, 2)
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """

    """ YOUR CODE HERE """
    EPOCHS = 40
    train_losses = []
    test_losses = []

    model = MADE(d, num_hidden_layers=80)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    p_dataset = preprocess_dataset(train_data, d)
    t_dataset = preprocess_dataset(test_data, d)
    dataloader = torch.utils.data.DataLoader(p_dataset, batch_size=128, shuffle=True)

    test_losses.append(test(dataloader, model, d))
    for epoch in range(EPOCHS):
      train_losses += train_epoch(dataloader, model, d, optimizer)
      test_losses.append(test(dataloader, model, d))
      print(test_losses[-1])
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    print(train_losses[-1])
    distribution = get_distribution(model, d)
    #print(distribution)
    return train_losses, test_losses, distribution

q2_save_results(1, 'a', q2_a)
