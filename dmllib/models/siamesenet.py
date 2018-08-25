# -*- coding: utf-8 -*-
import torch.nn as nn
from embeddingnet import EmbeddingNet


class SiameseNet(nn.Module):
    def __init__(self, in_ch, l2=True):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(in_ch, l2)

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2

    def embedding(self, x):
        return self.embedding_net(x)
