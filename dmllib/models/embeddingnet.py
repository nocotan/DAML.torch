# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, in_ch, l2=True):
        super(EmbeddingNet, self).__init__()
        self.in_ch = in_ch
        self.l2 = l2

        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h))
        h = F.leaky_relu(self.fc2(h))
        h = self.fc3(h)

        if self.l2:
            h /= h.pow(2).sum(1, keepdim=True).sqrt()

        return h
