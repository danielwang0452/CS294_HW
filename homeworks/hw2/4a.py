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

class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.n_channels = n_channels
        self.initialized = False

    def forward(self, x):
        N, C, H, W = x.shape
        if not self.initialized:
            self.shift.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
            self.log_scale.data = - torch.log(
                torch.std(x.permute(1, 0, 2, 3).reshape(self.n_channels, -1), dim=1).reshape(1, self.n_channels, 1,
                                                                                                 1))
            self.initialized = True
            result = x * torch.exp(self.log_scale) + self.shift
        out = x * torch.exp(self.log_scale) + self.shift
        #print(out.mean())
        #return x, torch.zeros((1), dtype=dtype).to(device)
        return out, self.log_scale

    def invert(self, y):
        return (y - self.shift) * torch.exp(-self.log_scale)

class AffineCoupling(nn.Module):
    def __init__(self, mask, in_channels, out_channels):
        super().__init__()
        self.mask = mask
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.ResNet = ResNet(in_channels, out_channels)

    def spatial_mask(self, shape, flip_mask): # 1 if sum of coords is odd
        # shape [b, c, h, w]
        N, C, H, W = shape
        mask = np.zeros(shape)
        for h in range(H):
           for w in range(W):
                if (h+w) % 2 == 1:
                   mask[:, :, h, w] = 1
        if flip_mask == True:
            mask = -mask + 1
        return torch.tensor(mask, dtype=dtype).to(device)

    def channel_mask(self, shape, flip_mask): # first half of channels = 1
        # shape [b, c*4, h//2, w//2]
        N, C, H, W = shape
        mask = np.zeros(shape)
        mask[:, :C//2, :, :] = 1
        if flip_mask == True:
            mask = -mask + 1
        return torch.tensor(mask, dtype=dtype).to(device)

    def forward(self, x):
        N, C, H, W = x.shape
        self.mask = self.mask[:N, :, :, :]
        x1 = x*self.mask
        x2 = x*(1-self.mask)
        scale, shift = torch.chunk(self.ResNet(x1), 2, dim=1) # same shape as x
        scale = scale*(1-self.mask)
        scale = self.scale*torch.tanh(scale) + self.scale_shift
        #print(scale.mean())
        shift = shift*(1-self.mask)
        y2 = x2 * scale.exp() + shift
        return x1 + y2, scale # x1 = y1, so no need to define y1

    def invert(self, y):
        N, C, H, W = y.shape
        self.mask = self.mask.repeat((10, 1, 1, 1))
        self.mask = self.mask[:N, :, :, :]
        #print(y.shape)
        #print(self.mask.shape)
        y1 = y*self.mask
        y2 = y*(1-self.mask)
        scale, shift = torch.chunk(self.ResNet(y1), 2, dim=1)
        scale = scale * (1 - self.mask)
        scale = self.scale * torch.tanh(scale) + self.scale_shift
        shift = shift * (1 - self.mask)
        x2 = (y2-shift)*(-1*scale).exp()
        return y1 + x2

class ChannelCoupling(nn.Module):
    def __init__(self, flip_mask, in_channels, out_channels):
        super().__init__()
        self.ResNet = ResNet(in_channels, out_channels)
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
        log_det = scale
        if self.flip_mask == False:
            return torch.cat((x1, y2), dim=1), \
                torch.cat((torch.zeros_like(log_det, dtype=dtype).to(device), log_det), dim=1) # x1 = y1, so no need to define y1
        else:
            return torch.cat((y2, x1), dim=1), \
                torch.cat((log_det, torch.zeros_like(log_det, dtype=dtype).to(device)), dim=1)
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

class ScaleGroup(nn.Module):
    def __init__(self, N, C, H, W):
        super().__init__()
        self.shape = (N, C, H, W)
        self.SpatialCoupling1 = AffineCoupling(self.spatial_mask(flip_mask=False),
                                               in_channels=C,
                                               out_channels=2*C)
        self.ActNorm1 = ActNorm(C)
        self.SpatialCoupling2 = AffineCoupling(self.spatial_mask(flip_mask=True),
                                               in_channels=C,
                                               out_channels=2*C)
        self.ActNorm2 = ActNorm(C)
        self.SpatialCoupling3 = AffineCoupling(self.spatial_mask(flip_mask=False),
                                               in_channels = C,
                                               out_channels = 2 * C)
        self.ActNorm3 = ActNorm(C)
        self.ChannelCoupling1 = ChannelCoupling(flip_mask=False,
                                               in_channels=C*2,
                                               out_channels=4 * C)
        self.ActNorm4 = ActNorm(4 * C)
        self.ChannelCoupling2 = ChannelCoupling(flip_mask=True,
                                               in_channels=C*2,
                                               out_channels=4 * C)
        self.ActNorm5 = ActNorm(4 * C)
        self.ChannelCoupling3 = ChannelCoupling(flip_mask=False,
                                               in_channels=C*2,
                                               out_channels=4 * C)
        self.ActNorm6 = ActNorm(4 * C)

    def spatial_mask(self, flip_mask): # 1 if sum of coords is odd
        # shape [b, c, h, w]
        N, C, H, W = self.shape
        mask = np.zeros(self.shape)
        for h in range(H):
           for w in range(W):
                if (h+w) % 2 == 1:
                   mask[:, :, h, w] = 1
        if flip_mask == True:
            mask = -mask + 1
        return torch.tensor(mask, dtype=dtype).to(device)

    def channel_mask(self, flip_mask): # first half of channels = 1
        # shape [b, c*4, h//2, w//2]
        N, C, H, W = self.shape
        C = C*4
        H = H//2
        W = W//2
        mask = np.zeros((N, C, H, W))
        mask[:, :C//2, :, :] = 1
        if flip_mask == True:
            mask = -mask + 1
        return torch.tensor(mask, dtype=dtype).to(device)

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

    def forward(self, y):
        log_det = torch.zeros_like(y, dtype=dtype).to(device)
        y, y_det = self.SpatialCoupling1(y)
        y, a_det = self.ActNorm1(y)
        log_det += y_det
        log_det += a_det

        y, y_det = self.SpatialCoupling2(y)
        y, a_det = self.ActNorm2(y)
        log_det += y_det
        log_det += a_det

        y, y_det = self.SpatialCoupling3(y)
        y, a_det = self.ActNorm3(y)
        log_det += y_det
        log_det += a_det

        y, log_det = self.squeeze(y), self.squeeze(log_det)

        y, y_det = self.ChannelCoupling1(y)
        y, a_det = self.ActNorm4(y)
        log_det += y_det
        log_det += a_det

        y, y_det = self.ChannelCoupling2(y)
        y, a_det = self.ActNorm5(y)
        log_det += y_det
        log_det += a_det

        y, y_det = self.ChannelCoupling3(y)
        y, a_det = self.ActNorm6(y)
        log_det += y_det
        log_det += a_det

        y, z = torch.chunk(y, 2, dim=1)
        return y, z, log_det

    def invert(self, y):
        y = self.ActNorm6.invert(y)
        y = self.ChannelCoupling3.invert(y)

        y = self.ActNorm5.invert(y)
        y = self.ChannelCoupling2.invert(y)

        y = self.ActNorm4.invert(y)
        y = self.ChannelCoupling1.invert(y)

        y = self.unsqueeze(y)
        y = self.ActNorm3.invert(y)
        y = self.SpatialCoupling3.invert(y)

        y = self.ActNorm2.invert(y)
        y = self.SpatialCoupling2.invert(y)

        y = self.ActNorm1.invert(y)
        y = self.SpatialCoupling1.invert(y)
        return y

class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_dist = Normal(torch.tensor(0.), torch.tensor(1.))
        # network architecture
        self.scale1 = ScaleGroup(64, 3, 32, 32)
        self.scale2 = ScaleGroup(64, 6, 16, 16)
        self.scale3 = ScaleGroup(64, 12, 8, 8)

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
        y = self.scale3.invert(z3)
        y = self.scale2.invert(torch.cat((y, z2), dim=1))
        y = self.scale1.invert(torch.cat((y, z1), dim=1))
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
        # print('a')
        # print((z1_prob.sum(dim=[1, 2, 3]) + log_det_1.sum(dim=(1, 2, 3))).mean()/(3*32*32))
        # print((z2_prob.sum(dim=[1, 2, 3]) + log_det_2.sum(dim=(1, 2, 3))).mean() / (3 * 32 * 32))
        # print((z3_prob.sum(dim=[1, 2, 3]) + log_det_3.sum(dim=(1, 2, 3))).mean() / (3 * 32 * 32))
        return z1_prob.sum(dim=[1, 2, 3]) + log_det_1.sum(dim=(1, 2, 3)) \
             + z2_prob.sum(dim=(1, 2, 3)) + log_det_2.sum(dim=(1, 2, 3)) \
             + z3_prob.sum(dim=(1, 2, 3)) + log_det_3.sum(dim=(1, 2, 3))
        # return (z1_prob + z2_prob + z3_prob).mean() + (log_det_1+log_det_2+log_det_3).mean()
    def loss(self, x):
        return -self.log_prob(x) # avg across batch

    def sample(self, N, C, H, W, image):
        # N is num samples
        mean = torch.zeros(N, C, H, W)
        std = torch.ones(N, C, H, W)
        z = torch.normal(mean=mean, std=std).to(device)
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
                #break

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
                '''
                z1, z2, z3, _, _, _ = model.forward(x)
                x0 = model.invert(z1, z2, z3)
                x0 = model.preprocess(x0, reverse=True, dequantize=False)
                print(x_init - x0)
                '''
                N, C, H, W = batch.shape
                log_prob = (model.log_prob(x))
                loss = -torch.mean(log_prob + x_det) / (C * H * W)

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


def q4_a(train_data, test_data):
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

    #state_dict = torch.load(f'4b_models/epoch120.pth', map_location='mps')
    #model.load_state_dict(state_dict)

    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    EPOCHS = 2
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # train
    train_losses, test_losses = train(model, train_loader, test_loader, EPOCHS, optimizer)
    # sample
    # test image
    for batch in train_loader:
        image = torch.tensor(batch, dtype=dtype).to(device)
        image, x_det = model.preprocess(image, reverse=False, dequantize=True)
        break
    samples = model.sample(64, 3, 32, 32, image)
    interpolations = np.transpose(interpolate(model, test_loader), axes=[0, 2, 3, 1])
    return train_losses[2:], test_losses, samples, interpolations

q3_save_results(q4_a, 'bonus_a')
# filters:blocks:epochs:params:loss
#32:4:5:504082:0.6029
#64:4:5:1909170:0.5644
#64:8:5:3719090:0.5408
#128:8:5:14647026:0.3583
#64:8:5:3719090: