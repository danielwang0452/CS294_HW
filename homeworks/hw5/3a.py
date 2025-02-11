from deepul.hw4_helper import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'mps'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

def q3_a(images, vae):
    """
    images: (1000, 32, 32, 3) numpy array in [0, 1], the images to pass through the encoder and decoder of the vae
    vae: a vae model, trained on the relevant dataset

    Returns
    - a numpy array of size (50, 2, 32, 32, 3) of the decoded image in [0, 1] consisting of pairs
      of real and reconstructed images
    - a float that is the scale factor
    """

    """ YOUR CODE HERE """
    # reconstruct images
    vae.to(device)
    images_in = images[:50]
    images_in = 2*(images_in.transpose((0, 3, 1, 2))-0.5)
    encoded_images = vae.encode(images_in) # shape (N, 4, 8, 8)
    decoded_images = vae.decode(encoded_images) # shape (N, 3, 32, 32)
    decoded_images = decoded_images.cpu().numpy().transpose((0, 2, 3, 1))/2+0.5
    decoded_images = np.expand_dims(decoded_images, 1)
    autoencoded_images = np.concatenate((np.expand_dims(images[:50], 1), decoded_images), axis=1)
    # compute scale factor
    images_in = 2 * (images.transpose((0, 3, 1, 2)) - 0.5)
    encoded_images = vae.encode(images_in)  # shape (N, 4, 8, 8)
    scale_factor = np.std(encoded_images.cpu().numpy().flatten())
    return autoencoded_images, scale_factor

q3a_save_results(q3_a)