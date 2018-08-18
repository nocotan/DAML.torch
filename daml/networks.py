# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_ch=512*3):
        super(Generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_ch, 512, kernel_size=2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv5 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)

    def forward(self, x):
        h = F.relu(self.dconv1(x))
        h = F.relu(self.dconv2(h))
        h = F.relu(self.dconv3(h))
        h = F.relu(self.dconv4(h))
        out = F.relu(self.dconv5(h))

        return out


class FeatureExtractor(nn.Module):
    def __init__(self, extractor, intermediate_size=512*8*8, out_ch=256):
        super(FeatureExtractor, self).__init__()
        self.intermediate_size = intermediate_size

        self.extractor = extractor
        self.fc1_1 = nn.Linear(intermediate_size, int(intermediate_size/2))
        self.fc2_1 = nn.Linear(intermediate_size, int(intermediate_size/2))
        self.fc3_1 = nn.Linear(intermediate_size, int(intermediate_size/2))
        self.fc1_2 = nn.Linear(int(intermediate_size/2), out_ch)
        self.fc2_2 = nn.Linear(int(intermediate_size/2), out_ch)
        self.fc3_2 = nn.Linear(int(intermediate_size/2), out_ch)

    def forward(self, x, y, z):
        batch_size = x.size(0)

        embedded_x = self.extractor(x)
        embedded_y = self.extractor(y)
        embedded_z = self.extractor(z)

        h_x = F.relu(self.fc1_1(embedded_x.view(batch_size, -1)))
        h_x = F.relu(self.fc1_2(h_x))

        h_y = F.relu(self.fc2_1(embedded_y.view(batch_size, -1)))
        h_y = F.relu(self.fc2_2(h_y))

        h_z = F.relu(self.fc3_1(embedded_z.view(batch_size, -1)))
        h_z = F.relu(self.fc3_2(h_z))

        dist_a = F.pairwise_distance(h_x, h_y, 2)
        dist_b = F.pairwise_distance(h_x, h_z, 2)

        return (dist_a, dist_b), (embedded_x, embedded_y, embedded_z)
