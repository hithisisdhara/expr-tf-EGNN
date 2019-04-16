# -*- coding: utf-8 -*-
"""
PyTorch models.
"""
import torch

class DCGANGenerator(torch.nn.Module):
    """DCGAN generator model

    Args:

      in_channels (int): Number of input (latent) channels

      out_channels (int): Number of output (image) channels. Default to 3

      image_size (int): Size of the generated image. Default to 64.

      min_hidden_channels (int): Number of channels of the last hidden
        convolutional layer, which is also the smallest number among
        all hidden convolutional layers. Default to 128.

      hidden_activation (callable): Activation layer or function
        which will be applied to results of each hidden layer. Default
        to LeakyReLU(inplace=True).

      out_activation (callable): Activation layer or function which
        will be applied to output. Default to Tanh().

    Examples:

    >>> G = DCGANGenerator(100, 3, 64, 128)
    >>> len(G.blocks)
    5
    >>> G.blocks[0][0].out_channels
    1024
    >>> G.blocks[3][0].out_channels
    128
    >>> Z = torch.ones(10, 100)
    >>> X = G(Z)
    >>> X.shape
    torch.Size([10, 3, 64, 64])
    >>> DCGANGenerator(100, 3, 28)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> DCGANGenerator(100, 3, 28.3)
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    def __init__(self, in_channels, out_channels=3, image_size=64,
                 min_hidden_channels=128,
                 hidden_activation=torch.nn.LeakyReLU(inplace=True),
                 out_activation=torch.nn.Tanh()):
        super().__init__()

        if not isinstance(image_size, int):
            raise TypeError('image_size must be an int.')
        if image_size <= 4 or image_size & image_size-1 != 0:
            raise ValueError('image_size must be a power of 2 greater than 4')
        nblocks = 1
        sz = image_size
        while sz > 4:
            nblocks += 1
            sz /= 2
        assert sz == 4 and nblocks >= 2
        layers = []
        C = min_hidden_channels * (2 ** (nblocks-2))
        IC, OC = in_channels, out_channels

        blocks = []
        # input block
        block = []
        block.append(torch.nn.ConvTranspose2d(IC, C, 4, 1, 0, bias=False))
        block.append(torch.nn.BatchNorm2d(C))
        if hidden_activation is not None:
            block.append(hidden_activation)
        blocks.append(torch.nn.Sequential(*block))
        
        # hidden blocks
        for _ in range(nblocks-2):
            block = []
            block.append(torch.nn.ConvTranspose2d(C,C//2,4,2,1,bias=False))
            block.append(torch.nn.BatchNorm2d(C//2))
            if hidden_activation is not None:
                block.append(hidden_activation)
            C //= 2
            blocks.append(torch.nn.Sequential(*block))
        assert C == min_hidden_channels

        # output block
        block = []
        block.append(torch.nn.ConvTranspose2d(C, OC, 4, 2, 1, bias=True))
        if out_activation is not None:
            block.append(out_activation)
        blocks.append(torch.nn.Sequential(*block))
        self.blocks = torch.nn.Sequential(*blocks)
                          

    def forward(self, Z):
        if Z.dim() == 2:
            Z = Z.unsqueeze(2).unsqueeze(3)
        return self.blocks(Z)


class DCGANDiscriminator(torch.nn.Module):
    """DCGAN discriminator model

    Args:

      in_channels (int): Number of input (latent) channels

      out_channels (int): Number of output (image) channels. Default to 3

      image_size (int): Size of the generated image. Default to 64.

      min_hidden_channels (int): Number of channels of the first hidden
        convolutional layer, which is also the smallest number among
        all hidden convolutional layers. Default to 128.

      hidden_activation (callable): Activation layer or function
        which will be applied to results of each hidden layer. Default
        to LeakyReLU(inplace=True).

      out_activation (callable): Activation layer or function which
        will be applied to output. Default to Sigmoid().

    Examples:

    >>> D = DCGANDiscriminator(1, 1, 64, 128)
    >>> len(D.blocks)
    5
    >>> D.blocks[1][0].in_channels
    128
    >>> D.blocks[3][0].out_channels
    1024
    >>> X = torch.rand(10, 1, 64, 64)
    >>> Y = D(X)
    >>> Y.shape
    torch.Size([10])
    >>> DCGANDiscriminator(1, 1, 28)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> DCGANDiscriminator(1, 1, 28.3)
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    def __init__(self, in_channels, out_channels=1, image_size=64,
                 min_hidden_channels=128,
                 hidden_activation=torch.nn.LeakyReLU(inplace=True),
                 out_activation=torch.nn.Sigmoid()):
        super().__init__()

        if not isinstance(image_size, int):
            raise TypeError('image_size must be an int.')
        if image_size <= 4 or image_size & image_size-1 != 0:
            raise ValueError('image_size must be a power of 2 greater than 4')
        nblocks = 1
        sz = image_size
        while sz > 4:
            nblocks += 1
            sz /= 2
        assert sz == 4 and nblocks >= 2
        layers = []
        IC, OC, C = in_channels, out_channels, min_hidden_channels
        layers = []

        blocks = []
        # input block
        block = []
        block.append(torch.nn.Conv2d(IC, C, 4, 2, 1, bias=True))
        if hidden_activation is not None:
            block.append(hidden_activation)
        blocks.append(torch.nn.Sequential(*block))
        
        # hidden blocks
        for _ in range(nblocks-2):
            block = []
            block.append(torch.nn.Conv2d(C, C*2, 4, 2, 1, bias=False))
            block.append(torch.nn.BatchNorm2d(C*2))
            if hidden_activation is not None:
                block.append(hidden_activation)
            blocks.append(torch.nn.Sequential(*block))
            C *= 2
        
        # output blocks
        block = []
        block.append(torch.nn.Conv2d(C, OC, 4, 1, 0, bias=True))
        if out_activation is not None:
            block.append(out_activation)
        blocks.append(torch.nn.Sequential(*block))
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, X):
        h = self.blocks(X)
        if h.size(3) == 1:
            h = h.squeeze(3)
        if h.size(2) == 1:
            h = h.squeeze(2)
        if h.size(1) == 1:
            h = h.squeeze(1)
        return h
