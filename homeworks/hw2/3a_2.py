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
        # return torch.tanh(h) + x


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
        # out = torch.tanh(out)
        return out


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
            result = x * torch.exp(self.log_scale) + self.shift
        return x * torch.exp(self.log_scale) + self.shift, self.log_scale.squeeze()

    def invert(self, y):
        return (y - self.shift) * torch.exp(-self.log_scale)


class ActNorm2(nn.Module):
    def __init__(self):
        super().__init__()

    def param_init(self, x):
        # x is (N, C, H, W)
        N, C, H, W = x.shape
        self.weight = []
        self.bias = []
        for c in range(C):
            mean = x[:, c, :, :].mean()
            var = ((x - mean * torch.ones_like(x)) ** 2).mean()
            self.weight.append(1 / var)
            self.bias.append(mean)
        self.weight = nn.Parameter(torch.tensor(self.weight, dtype=dtype), requires_grad=True).to(device)
        self.bias = nn.Parameter(torch.tensor(self.bias, dtype=dtype), requires_grad=True).to(device)

    def forward(self, x):
        if init_actnorm == True:
            self.param_init(x)
        x = x.permute((0, 2, 3, 1))  # N, H, W, C
        # weight and bias are size (C)
        return (self.weight * x - self.bias).permute(0, 3, 1, 2), F.relu(self.weight).log()  # back to N, C, H, W

    def invert(self, y):
        y = y.permute((0, 2, 3, 1))
        return ((y + self.bias) / (self.weight)).permute(0, 3, 1, 2)


class AffineCoupling(nn.Module):
    def __init__(self, mask_type, flip_mask):
        super().__init__()
        if mask_type == 'spatial':
            shape = (64, 3, 32, 32)
            self.mask = self.spatial_mask(shape, flip_mask)
            self.ResNet = ResNet(3, 6, n_filters=64, n_blocks=8)
        elif mask_type == 'channel':
            shape = (64, 12, 16, 16)
            self.mask = self.channel_mask(shape, flip_mask)
            self.ResNet = ResNet(12, 24, n_filters=64, n_blocks=8)

        self.scale = nn.Parameter(torch.ones(1), requires_grad=True).to(device)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)

    def spatial_mask(self, shape, flip_mask):  # 1 if sum of coords is odd
        # shape [b, c, h, w]
        N, C, H, W = shape
        mask = np.zeros(shape)
        for h in range(H):
            for w in range(W):
                if (h + w) % 2 == 1:
                    mask[:, :, h, w] = 1
        if flip_mask == True:
            mask = -mask + 1
        return torch.tensor(mask, dtype=dtype).to(device)

    def forward(self, x):
        N, C, H, W = x.shape
        self.mask = self.mask[:N, :, :, :]
        x1 = x * self.mask
        x2 = x * (1 - self.mask)
        scale, shift = torch.chunk(self.ResNet(x1), 2, dim=1)  # same shape as x

        scale = scale * (1 - self.mask)
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        shift = shift * (1 - self.mask)
        y2 = x * (1 - self.mask) * scale.exp() + shift
        # print(self.scale_shift.grad)
        # print(scale)
        log_det = scale

        return x1 + y2, log_det  # x1 = y1, so no need to define y1

    def invert(self, y):
        N, C, H, W = y.shape
        self.mask = self.mask.repeat((10, 1, 1, 1))
        self.mask = self.mask[:N, :, :, :]
        y1 = y * self.mask
        y2 = y * (1 - self.mask)
        scale, shift = torch.chunk(self.ResNet(y1), 2, dim=1)
        scale = scale * (1 - self.mask)
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        shift = shift * (1 - self.mask)
        x2 = (y2 - shift) * (-1 * scale).exp()
        return y1 + x2

class ChannelCoupling(nn.Module):
    def __init__(self, mask_type, flip_mask):
        super().__init__()
        self.ResNet = ResNet(6, 12, n_filters=64, n_blocks=8)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True).to(device)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)
        self.flip_mask = flip_mask

    def forward(self, x):
        if self.flip_mask == False:
            x1, x2 = torch.chunk(x, 2, dim=1)
        else:
            x2, x1 = torch.chunk(x, 2, dim=1)

        scale, shift = torch.chunk(self.ResNet(x1), 2, dim=1)  # same shape as x
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        y2 = x2 * scale.exp() + shift
        # print(self.scale_shift.grad)
        # print(scale)
        log_det = scale/2
        if self.flip_mask == False:
            return torch.cat((x1, y2), dim=1), log_det  # x1 = y1, so no need to define y1
        else:
            return torch.cat((y2, x1), dim=1), log_det

    def invert(self, y):
        if self.flip_mask == False:
            y1, y2 = torch.chunk(y, 2, dim=1)
        else:
            y2, y1 = torch.chunk(y, 2, dim=1)

        scale, shift = torch.chunk(self.ResNet(y1), 2, dim=1)
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        x2 = (y2 - shift) * (-1 * scale).exp()
        if self.flip_mask == False:
            return torch.cat((y1, x2), dim=1)
        else:
            return torch.cat((x2, y1), dim=1)


class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_dist = Normal(torch.tensor(0.), torch.tensor(1.))
        # network architecture
        self.group1 = nn.ModuleList([])
        for i in range(4):
            flip_mask = True
            if i % 2 == 0:  # start with no flipping
                flip_mask = False
            self.group1.append(AffineCoupling(mask_type='spatial', flip_mask=flip_mask))
            self.group1.append(ActNorm(n_channels=3))
        self.group2 = nn.ModuleList([])
        for i in range(3):
            flip_mask = True
            if i % 2 == 0:  # start with no flipping
                flip_mask = False
            self.group2.append(ChannelCoupling(mask_type='channel', flip_mask=flip_mask))
            self.group2.append(ActNorm(n_channels=12))
        self.group3 = nn.ModuleList([])
        for i in range(3):
            flip_mask = True
            if i % 2 == 0:  # start with no flipping
                flip_mask = False
            self.group3.append(AffineCoupling(mask_type='spatial', flip_mask=flip_mask))
            self.group3.append(ActNorm(n_channels=3))

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
        N, C, H, W = x.shape
        y = x
        log_det = torch.zeros(x.shape[0], dtype=dtype).to(device)  # batch size
        actnorm_det_g13 = torch.zeros(x.shape[1], dtype=dtype).to(device)  # num channels
        actnorm_det_g2 = torch.zeros(x.shape[1] * 4, dtype=dtype).to(device)  # num channels
        # group 1
        for layer in self.group1:
            if isinstance(layer, AffineCoupling):
                y, y_det = layer(y)  # y, y_det are same shape
                log_det += y_det.mean(dim=(1, 2, 3))
            if isinstance(layer, ActNorm):
                y, y_det = layer(y)
                actnorm_det_g13 += y_det
        y = self.squeeze(y)
        # group 2
        for layer in self.group2:
            if isinstance(layer, ChannelCoupling):
                y, y_det = layer(y)  # y, y_det are same shape
                log_det += y_det.mean(dim=(1, 2, 3))
            if isinstance(layer, ActNorm):
                y, y_det = layer(y)
                actnorm_det_g2 += y_det
        y = self.unsqueeze(y)
        # group 3
        for layer in self.group3:
            if isinstance(layer, AffineCoupling):
                y, y_det = layer(y)  # y, y_det are same shape
                log_det += y_det.mean(dim=(1, 2, 3))
            if isinstance(layer, ActNorm):
                y, y_det = layer(y)
                actnorm_det_g13 += y_det

        return y, log_det, actnorm_det_g2, actnorm_det_g13  # both same shape

    def init_actnorm(self, minibatch):
        self.forward(minibatch, init_actnorm=True)

    def invert(self, z):
        # loop through every layer in reverse
        y = z
        for layer in reversed(self.group3):
            y = layer.invert(y)
        y = self.squeeze(y)
        for layer in reversed(self.group2):
            y = layer.invert(y)
        y = self.unsqueeze(y)
        for layer in reversed(self.group1):
            y = layer.invert(y)
        return y

    def log_prob(self, x):
        y, log_det, actnorm_det_g2, actnorm_det_g13 = self.forward(x)
        z = self.base_dist.log_prob(y)
        # z is shape (N, C, H, W), log_det is shape(N) and actnorm_det is shape (C)
        # print(z.mean())
        # print(self.base_dist.log_prob(z).mean())
        # print(log_det.mean() + actnorm_det_g2.mean() + actnorm_det_g13.mean())
        return log_det.mean() + z.mean() + actnorm_det_g2.mean() + actnorm_det_g13.mean()

    def loss(self, x):
        return -self.log_prob(x)  # avg across batch

    def sample(self, N, C, H, W):
        # N is num samples
        mean = torch.zeros(N, C, H, W)
        std = torch.ones(N, C, H, W)
        z = torch.normal(mean=mean, std=std).to(device)
        x = self.invert(z)
        x = self.preprocess(x, reverse=True, dequantize=False)
        x = torch.permute(x, (0, 2, 3, 1))  # shape (N, H, W, C)
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
            return logit, torch.mean(log_det)


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

                log_prob = (model.log_prob(x))
                loss = -(log_prob + x_det)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())

                train_losses.append(loss.item())
                break
        print(f'train loss: {loss}')
        test_losses.append(test(model, test_dataloader))
        model_path = f'3a_models/epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)
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
                '''
                z, _, _, _ = model.forward(x)
                x0 = model.invert(z)
                x1 = model.preprocess(x0, reverse=True, dequantize=False)
                print(x1-x_init)
                '''
                loss = -(model.log_prob(x) + x_det)
                epoch_losses.append(loss.item())
                break

        loss = np.array(epoch_losses).mean()
        print(f'test loss: {loss}')
        return loss

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
        latent_images, _, _, _ = model.forward(logit_actual_images)
        latents = []
        for i in range(0, actual_images.shape[0], 2):
            a = latent_images[i:i+1]
            b = latent_images[i + 1:i+2]
            diff = (b - a)/5.0
            latents.append(a)
            for j in range(1, 5):
                latents.append(a + diff * float(j))
            latents.append(b)
        latents = torch.cat(latents, dim=0)
        logit_results = model.invert(latents)
        results = model.preprocess(logit_results, reverse=True)
        return results.cpu().numpy()

def q3_a(train_data, test_data):
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
    model = RealNVP()
    model.to(device)

    # state_dict = torch.load(f'3a_models/epoch35.pth', map_location='mps')
    # model.load_state_dict(state_dict)

    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    EPOCHS = 1
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # train
    train_losses, test_losses = train(model, train_loader, test_loader, EPOCHS, optimizer)
    # sample
    samples = model.sample(64, 3, 32, 32)
    interpolations = np.transpose(interpolate(model, test_loader), axes=[0, 2, 3, 1])
    return train_losses, test_losses, samples, interpolations


q3_save_results(q3_a, 'a')
# filters:blocks:epochs:params:loss
# 32:4:5:504082:0.6029
# 64:4:5:1909170:0.5644
# 64:8:5:3719090:0.5408
# 128:8:5:14647026:0.3583
# 64:8:5:3719090: