import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# need to set this before importing torch
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

class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.n_channels = n_channels
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.shift.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
            self.log_scale.data = - torch.log(
                torch.std(x.permute(1, 0, 2, 3).reshape(self.n_channels, -1), dim=1).reshape(1, self.n_channels, 1,
                                                                                                 1))
            self.initialized = True
        #return x, self.log_scale.squeeze()*0
        return x * torch.exp(self.log_scale) + self.shift, self.log_scale

    def invert(self, y):
        return (y - self.shift) * torch.exp(-self.log_scale)

class InvertibleConv(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        # generate rotation matrix
        Q, R = np.linalg.qr(np.random.randn(self.n_channels, self.n_channels))
        # init weights
        weight = torch.tensor(Q, dtype=dtype).unsqueeze(-1).unsqueeze(-1) # shape(C, C, 1, 1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        mean, std = torch.zeros((n_channels), dtype=dtype), torch.ones((n_channels), dtype=dtype)
        #self.bias = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((n_channels), dtype=dtype), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        out = F.conv2d(x, self.weight, bias=None)

        #np_matrix = (self.weight.squeeze()).detach().cpu().numpy()
        #det = torch.tensor(np.linalg.det(np_matrix), dtype=dtype).to(device)
        det = torch.linalg.det(self.weight.squeeze())
        log_det = torch.log(abs(det))/C # real number
        return out, log_det

    def invert(self, y):
        # y is (N, C, H, W), bias is (C) -> subtract bias from every
        y = (y.permute((0, 2, 3, 1)) - self.bias).permute((0, 3, 1, 2))
        inv_weight = torch.linalg.inv(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        out = F.conv2d(y, weight=inv_weight, bias=None)
        return out

class ResBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.net = nn.ModuleList([nn.Conv2d(n_filters, n_filters, kernel_size=1, padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(n_filters, n_filters, kernel_size=1, padding=0)])

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)
        return h + x
        #return torch.tanh(h) + x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_out, n_filters=128, n_blocks=8):
        super().__init__()
        self.net = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, padding=1)])
        for _ in range(n_blocks):
            self.net.append(ResBlock(n_filters))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(in_channels=n_filters, out_channels=n_out, kernel_size=3, padding=1))

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        #out = torch.tanh(out)
        return out
    
class AffineCoupling(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.ResNet = ResNet(out_channels, out_channels*2)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        scale, shift = torch.chunk(self.ResNet(x1), 2, dim=1)  # same shape as x
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        y2 = x2 * scale.exp() + shift
        log_det = scale
        return torch.cat((x1, y2), dim=1), \
                torch.cat((torch.zeros_like(log_det, dtype=dtype).to(device), log_det), dim=1) # x1 = y1, so no need to define y1

    def invert(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)

        scale, shift = torch.chunk(self.ResNet(y1), 2, dim=1)
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        x2 = (y2 - shift) * (-1 * scale).exp()
        return torch.cat((y1, x2), dim=1)


class ScaleGroup(nn.Module):
    def __init__(self, N, C, H, W, n_steps):
        super().__init__()
        self.shape = (N, C, H, W)
        self.n_steps = n_steps
        self.net = nn.ModuleList([])
        for n in range(n_steps):
            self.net.append(GlowStep(C*4))

    def squeeze(self, x):
        # C x H x W -> 4C x H/2 x W/2
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return x

    def unsqueeze(self, x):
        #  4C x H/2 x W/2  ->  C x H x W
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, H * 2, W * 2)
        return x

    def forward(self, x):
        log_det = torch.zeros_like(x, dtype=dtype).to(device)
        out, log_det = self.squeeze(x), self.squeeze(log_det)
        for i, block in enumerate(self.net):
            out, block_det = block(out)
            log_det += block_det
        y, z = torch.chunk(out, 2, dim=1)
        return y, z, log_det

    def invert(self, y):
        for block in reversed(self.net):
            y = block.invert(y)
        return self.unsqueeze(y)

class GlowStep(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.ActNorm = ActNorm(n_channels)
        self.Inv_conv = InvertibleConv(n_channels)
        self.AffineCoupling = AffineCoupling(n_channels // 2)

    def forward(self, x):
        log_det = torch.zeros_like(x, dtype=dtype).to(device)
        out, act_det = self.ActNorm(x)
        log_det += act_det
        out, conv_det = self.Inv_conv(out)
        log_det += conv_det
        out, affine_det = self.AffineCoupling(out)
        log_det += affine_det
        #print(conv_det.mean())
        #print(affine_det.mean())
        #print(act_det.mean())
        return out, log_det

    def invert(self, y):
        y = self.AffineCoupling.invert(y)
        y = self.Inv_conv.invert(y)
        y = self.ActNorm.invert(y)
        return y
    
class Glow(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_dist = Normal(torch.tensor(0.), torch.tensor(1.))
        # network architecture
        self.scale1 = ScaleGroup(64, 3, 32, 32, self.n_blocks)
        self.scale2 = ScaleGroup(64, 6, 16, 16, self.n_blocks)
        self.scale3 = ScaleGroup(64, 12, 8, 8, self.n_blocks)

    def forward(self, x):
        N, C, H, W = x.shape
        log_det = torch.zeros(x.shape[0], dtype=dtype).to(device)

        y1, z1, log_det_1 = self.scale1(x)
        y2, z2, log_det_2 = self.scale2(y1)
        y3, z3, log_det_3 = self.scale3(y2)
        #print('a')
        #print(log_det_3.sum(dim=(1, 2, 3)).mean()/(3*32*32))
        #print(log_det_2.sum(dim=(1, 2, 3)).mean())
        #print(log_det_3.sum(dim=(1, 2, 3)).mean())
        #return z1, z2, torch.cat((y3, z3), dim=1), log_det_1, log_det_2, log_det_3
        #return torch.cat((y1, z1), dim=1), log_det_1
        return z1, z2, torch.cat((y3, z3), dim=1), log_det_1, log_det_2, log_det_3

    def invert(self, z1, z2, z3):
        '''
        z3 = torch.normal(mean=torch.zeros((64, 48, 4, 4)).to(device),
                          std=torch.ones((64, 48, 4, 4)).to(device))
        z2 = torch.normal(mean=torch.zeros((64, 12, 8, 8)).to(device),
                          std=torch.ones((64, 12, 8, 8)).to(device))
        z1 = torch.normal(mean=torch.zeros((64, 6, 16, 16)).to(device),
                          std=torch.ones((64, 6, 16, 16)).to(device))
        '''
        y = self.scale3.invert(z3)
        y = self.scale2.invert(torch.cat((y, z2), dim=1))
        y = self.scale1.invert(torch.cat((y, z1), dim=1))
        #print(x-y)
        return y

    def log_prob(self, x):
        N, C, H, W = x.shape
        z1, z2, z3, log_det_1, log_det_2, log_det_3 = self.forward(x)
        #z1, z2, log_det_1, log_det_2 = self.forward(x)
        z1_prob = self.base_dist.log_prob(z1)
        z2_prob = self.base_dist.log_prob(z2)
        z3_prob = self.base_dist.log_prob(z3)
        #print(log_det_1.shape)
        #print(log_det_2.shape)
        #print(log_det_3.shape)
        #print((log_det_1.sum(dim=(1, 2, 3)) +
        #      log_det_2.sum(dim=(1, 2, 3)) +
        #      log_det_3.sum(dim=(1, 2, 3))).mean() / (C * H * W))
        #print((z1_prob.sum(dim=[1, 2, 3])
        #      + z2_prob.sum(dim=[1, 2, 3])
        #      + z3_prob.sum(dim=[1, 2, 3])).mean()/(3*32*32))
        # z is shape (N, C, H, W), log_det is shape(N) and actnorm_det is shape (C)
        #print('a')
        #print((z1_prob.sum(dim=[1, 2, 3]) + log_det_1.sum(dim=(1, 2, 3))).mean()/(3*32*32))
        #print((z2_prob.sum(dim=[1, 2, 3]) + log_det_2.sum(dim=(1, 2, 3))).mean() / (3 * 32 * 32))
        #print((z3_prob.sum(dim=[1, 2, 3]) + log_det_3.sum(dim=(1, 2, 3))).mean() / (3 * 32 * 32))
        return z1_prob.sum(dim=[1, 2, 3]) + log_det_1.sum(dim=(1, 2, 3)) \
             + z2_prob.sum(dim=(1, 2, 3)) + log_det_2.sum(dim=(1, 2, 3)) \
             + z3_prob.sum(dim=(1, 2, 3)) + log_det_3.sum(dim=(1, 2, 3))
        #return (z1_prob + z2_prob + z3_prob).mean() + (log_det_1+log_det_2+log_det_3).mean()
    def loss(self, x):
        return -self.log_prob(x) # avg across batch

    def sample(self, N, C, H, W):
        # N is num samples
        mean = torch.zeros(N, C, H, W)
        std = torch.ones(N, C, H, W)
        z3 = torch.normal(mean=torch.zeros((64, 48, 4, 4)).to(device),
                          std=torch.ones((64, 48, 4, 4)).to(device))
        z2 = torch.normal(mean=torch.zeros((64, 12, 8, 8)).to(device),
                          std=torch.ones((64, 12, 8, 8)).to(device))
        z1 = torch.normal(mean=torch.zeros((64, 6, 16, 16)).to(device),
                          std=torch.ones((64, 6, 16, 16)).to(device))
        x = self.invert(z1, z2, z3)
        # test
        #z1, z2, log_det_1, log_det_2 = self.forward(image)

        #x = self.invert(z1)

        x = self.preprocess(x, reverse=True, dequantize=False)
        x = torch.permute(x, (0, 2, 3, 1)) # shape (N, H, W, C)


        return x.cpu().detach().numpy()

    def preprocess(self, x, reverse=False, dequantize=True):
        if reverse:  # doesn't map back to [0, 4]
            x = 1.0 / (1 + torch.exp(-x))
            x -= 0.05
            x /= 0.9
            return x
        else:
            # dequantization
            if dequantize:
                x += torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(device)
            x /= 4.0

            # logit operation
            x *= 0.9
            x += 0.05
            logit = torch.log(x) - torch.log(1.0 - x)
            log_det = torch.nn.functional.softplus(logit) + torch.nn.functional.softplus(-logit) \
                      + torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(4.0))
            return logit, torch.sum(log_det, dim=[1, 2, 3])

def interpolate(model, dataloader):
    val_loader = dataloader
    model.eval()
    good = [5, 13, 16, 19, 22]
    indices = []
    for index in good:
        indices.append(index*2)
        indices.append(index*2+1)
    with torch.no_grad():
        actual_images = next(iter(val_loader))[indices].to(device)
        assert actual_images.shape[0] % 2 == 0
        logit_actual_images, _ = model.preprocess(actual_images.float(), dequantize=False)
        z1, z2, z3, _, _, _ = model.forward(logit_actual_images)
        latents1 = []
        latents2 = []
        latents3 = []
        '''
        for i in range(0, actual_images.shape[0], 2):
            a = latent_images[i:i+1]
            b = latent_images[i + 1:i+2]
            diff = (b - a)/5.0
            latents.append(a)
            for j in range(1, 5):
                latents.append(a + diff * float(j))
            latents.append(b)
        '''
        for i in range(0, actual_images.shape[0], 2):
            a1 = z1[i:i+1]
            a2 = z2[i:i + 1]
            a3 = z3[i:i + 1]
            b1 = z1[i + 1:i+2]
            b2 = z2[i + 1:i + 2]
            b3 = z3[i + 1:i + 2]
            diff1 = (b1 - a1)/5.0
            diff2 = (b2 - a2) / 5.0
            diff3 = (b3 - a3) / 5.0
            latents1.append(a1)
            latents2.append(a2)
            latents3.append(a3)
            for j in range(1, 5):
                latents1.append(a1 + diff1 * float(j))
                latents2.append(a2 + diff2 * float(j))
                latents3.append(a3 + diff3 * float(j))
            latents1.append(b1)
            latents2.append(b2)
            latents3.append(b3)
        latents1 = torch.cat(latents1, dim=0)
        latents2 = torch.cat(latents2, dim=0)
        latents3 = torch.cat(latents3, dim=0)
        logit_results = model.invert(latents1, latents2, latents3)
        results = model.preprocess(logit_results, reverse=True)
        return results.cpu().numpy()

def train(model, train_dataloader, test_dataloader, epochs, optimizer):
    train_losses = []
    test_losses = []
    test_losses.append(test(model, test_dataloader))
    for epoch in range(epochs):
        model.train()
        # print(epoch)
        for batch in train_dataloader:
            if batch.shape[0] == 64:
                x0 = torch.tensor(batch, dtype=dtype).to(device)
                x, x_det = model.preprocess(x0, reverse=False, dequantize=True)

                N, C, H, W = batch.shape
                log_prob = (model.log_prob(x))
                #print(x_det.mean() / (C*H*W))
                loss = -torch.mean(log_prob + x_det)/(C*H*W)
                train_losses.append(loss.item())
                #break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())



        print(f'train loss: {loss}')
        test_losses.append(test(model, test_dataloader))
        #model_path = f'4b_models/epoch{epoch+124}.pth'
        #torch.save(model.state_dict(), model_path)
    return np.array(train_losses), np.array(test_losses)

def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        epoch_losses = []
        for batch in dataloader:
            if batch.shape[0] == 64:
                x = torch.tensor(batch, dtype=dtype).to(device)
                x_init = x
                x, x_det = model.preprocess(x, reverse=False, dequantize=True)
                # test inverting

                z1, z2, z3, _, _, _ = model.forward(x)
                x0 = model.invert(z1, z2, z3)
                #print(x - x0)
                x0 = model.preprocess(x0, reverse=True, dequantize=False)
                #print(x_init - x0)

                N, C, H, W = batch.shape
                log_prob = (model.log_prob(x))
                loss = -torch.mean(log_prob + x_det) / (C * H * W)

                epoch_losses.append(loss.item())
                break
        loss = np.array(epoch_losses).mean()
        print(f'test loss: {loss}')
        return loss

def q4_b(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """
    """ YOUR CODE HERE """
    # data preprocessing
    train_data = np.transpose(train_data, axes=[0, 3, 1, 2]).astype(np.float32)  # NCHW 20000 x 3 x 32 x 32
    test_data = np.transpose(test_data, axes=[0, 3, 1, 2]).astype(np.float32)  # NCHW 6838 x 3 x 32 x 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=False)
    # model
    model = Glow(n_blocks=5)
    model.to(device)

    #state_dict = torch.load(f'4b_models/epoch30.pth', map_location='mps')
    #model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    EPOCHS = 75
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # train
    train_losses, test_losses = train(model, train_loader, test_loader, EPOCHS, optimizer)
    # sample
    samples = model.sample(64, 3, 32, 32)
    interpolations = np.transpose(interpolate(model, test_loader), axes=[0, 2, 3, 1])
    return train_losses, test_losses, samples, interpolations

q3_save_results(q4_b, 'bonus_b')
