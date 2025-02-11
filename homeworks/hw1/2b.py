from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'mps'

#-------------------------------------------------------
#visualize_q2a_data(dset_type=1)
#visualize_q2a_data(dset_type=2)

class MADE2(nn.Module):
  def __init__(self, data_dim, num_hidden_layers):
    super().__init__()
    self.d = data_dim
    self.num_hidden_layers = num_hidden_layers
    self.mask1 = create_mask(1, self.d)
    self.mask2 = create_mask(2, self.d)
    self.mask3 = create_mask(3, self.d)
    '''
    self.mask1 = torch.ones((2*self.d, self.d), dtype=torch.float32).to(device)
    self.mask2 = torch.ones((self.d, self.d), dtype=torch.float32).to(device)
    self.mask3 = torch.ones((self.d, 2*self.d), dtype=torch.float32).to(device)
    '''
    # initialise network params
    # input params
    self.input_weight = nn.Parameter(torch.empty((self.d*2, self.d)), requires_grad=True)
    self.input_bias = nn.Parameter(torch.empty(self.d), requires_grad=True)
    # hidden params
    self.weights = {}
    self.biases = {}
    for i in range(self.num_hidden_layers):
      setattr(self, f'weights_{i}', nn.Parameter(torch.empty((self.d, self.d)), requires_grad=True))
      nn.init.xavier_normal_(getattr(self, f'weights_{i}'))
      setattr(self, f'biases_{i}', nn.Parameter(torch.empty((self.d)), requires_grad=True))
      nn.init.normal_(getattr(self, f'biases_{i}'), mean=0, std=0.01)
    # output params
    self.output_weight = nn.Parameter(torch.empty((self.d, self.d*2)), requires_grad=True)
    self.output_bias = nn.Parameter(torch.empty(self.d*2), requires_grad=True)

    nn.init.xavier_normal_(self.input_weight)
    nn.init.normal_(self.input_bias, mean=0, std=0.01)
    nn.init.xavier_normal_(self.output_weight)
    nn.init.normal_(self.output_bias, mean=0, std=0.01)

  def forward(self, x):
    m = nn.ReLU()
    # input layer
    out = m(F.linear(x, (self.input_weight*self.mask1).T, self.input_bias))
    #print(self.input_weight)
    #print(self.input_bias)
    #print(out)
    # hidden layer
    for n in range(self.num_hidden_layers):
      out = m(F.linear(out, (getattr(self, f'weights_{n}')*self.mask2).T, getattr(self, f'biases_{n}')))
    out = F.linear(out, (self.output_weight*self.mask3).T, self.output_bias)
    return out

def train_epoch(dataloader, model, d, optimizer):
  '''
  :param dataloader: shape (N, H*W, 1)
  '''
  model.train()
  epoch_loss = []
  x_matrix = create_mask(4, d)
  for batch in dataloader:
    # batch is shape (N samples, len(x)=800)
    x = torch.tensor(batch, dtype=torch.float32).to(device)
    # get label
    y = torch.tensor(np.matmul(x.cpu().detach().numpy(), x_matrix), dtype=torch.long).to(device) # shape (N, H*W)
    N, M = y.shape
    logits = model.forward(x) # shape (N, 2*H*W)
    #print(logits)
    logits = logits.reshape((N, M, 2)).permute((0, 2, 1))
    loss = F.cross_entropy(logits, y)

    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(loss)
  return epoch_loss

def test(dataloader, model, d):
  model.eval()
  epoch_loss = []
  x_matrix = create_mask(4, d)
  with torch.no_grad():
    for batch in dataloader:
      # batch is shape (N samples, len(x)=800)
      x = torch.tensor(batch, dtype=torch.float32).to(device)
      # get label
      y = torch.tensor(np.matmul(x.cpu().detach().numpy(), x_matrix), dtype=torch.long).to(device)  # shape (N, H*W)
      N, M = y.shape
      logits = model.forward(x)  # shape (N, 2*H*W)
      # print(logits)
      logits = logits.reshape((N, M, 2)).permute((0, 2, 1))
      loss = F.cross_entropy(logits, y)

      epoch_loss.append(loss.item())

    return torch.mean(torch.tensor(epoch_loss))


def sample(model, d, H, W):
  model.eval()
  # initialise input
  input = []
  outputs = []
  for _ in range(2*d):
    input.append(0)
  # sample
  with torch.no_grad():
    for i in range(d): # autoregressively generate each pixel
      x = torch.tensor([input], dtype=torch.float32).to(device)
      logits = model.forward(x)  # shape (N, 2*H*W)
      probs = F.softmax(logits[:, 2*i:2*(i+1)])
      # print(logits)
      # append output to input
      pred = torch.multinomial(probs, 1, replacement=True)
      outputs.append(pred[0][0].item())
      if pred[0] == 0:
        input[2*i] == 1
      elif pred[0] == 1:
        input[2*i+1] == 1
    outputs = np.array(outputs).reshape((H, W, 1))
  return outputs # numpy array

def create_mask(type, d):
  '''
  :param type {1, 2, 3}
  type 1: for input layer
  type 2: for hidden layers
  type 3: for output layer
  type 4: matrix for converting x -> y
  :return:
  '''
  mask = []

  if type == 1:
    for i in range(2 * d):
      new_row = []
      for j in range(d):
        if j * 2 + 1 < i:
          new_row.append(1)
        else:
          new_row.append(0)
      mask.append(new_row)
    mask = torch.tensor(mask, requires_grad=False).type(torch.float32).to(device)


  elif type == 2:
    for i in range(d):
      new_row = []
      for j in range(d):
        if j <= i:
          new_row.append(1)
        else:
          new_row.append(0)
      mask.append(new_row)
    mask = torch.tensor(mask, requires_grad=False).type(torch.float32).to(device)

  elif type == 3:
    for i in range(d):
      new_row = []
      for j in range(2 * d):
        if j <= i * 2 + 1:
          new_row.append(1)
        else:
          new_row.append(0)
      mask.append(new_row)
    mask = torch.tensor(mask, requires_grad=False).type(torch.float32).to(device)

  elif type == 4:
    for i in range(2 * d):
      new_row = []
      for j in range(d):
        if j * 2 <= i and i % 2 != 0 and j * 2 + 1 == i:
          new_row.append(1)
        else:
          new_row.append(0)
      mask.append(new_row)
    mask = np.array(mask)

  return mask

def preprocess_dataset(dataset, d):
  '''
  :param dataset: A (n_train, H, W) uint8 numpy array of binary images with values in {0, 1}
  :return: reshapes dataset by flattening image (n_train, H*W), then one hot encodes x, add y
  shape (N samples, len(x)=800) where len x = 2*H*W is one hot encoded & concatenated,
                                  len y = 400, contains 0 or 1 for each x-pair
  '''
  n_train, H, W = dataset.shape
  dataset = dataset.reshape(n_train, H*W)
  new_dataset = []
  for sample in dataset:
    encoded_x = [] # concatenates pixels into dim 800 vector
    for pixel in sample:
      if pixel == 0:
        encoded_x += [1, 0]
      elif pixel == 1:
        encoded_x += [0, 1]
    new_dataset.append(encoded_x)
  new_dataset = np.array(new_dataset)
  return new_dataset

def q2_b(train_data, test_data, image_shape, dset_id):
  """
  train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
  test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
  image_shape: (H, W), height and width of the image
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
  """

  """ YOUR CODE HERE """
  train_data = np.squeeze(train_data, axis=3)
  test_data = np.squeeze(test_data, axis=3)
  H, W = image_shape
  d = H*W
  EPOCHS = 20
  train_losses = []
  test_losses = []
  samples = []

  model = MADE2(d, num_hidden_layers=30)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

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

  for m in range(100):
    print(m)
    samples.append(sample(model, d, H, W))
  samples = np.array(samples)

  return train_losses, test_losses, samples

q2_save_results(2, 'b', q2_b)