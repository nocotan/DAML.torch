# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Updater(object):
    def __init__(self, **kwargs):
        self.extractor, self.generator = kwargs.pop("models")
        self.e_optim, self.g_optim = kwargs.pop("optimizers")
        self.alpha = kwargs.pop("alpha")
        self.lmd = kwargs.pop("lmd")

        self.mse = nn.MSELoss()

    def update(self, triplet):
        alpha = self.alpha
        lmd = self.lmd

        x1, x2, x3 = triplet
        x1 = x1.repeat(1, 3, 1, 1)
        x2 = x2.repeat(1, 3, 1, 1)
        x3 = x3.repeat(1, 3, 1, 1)

        dist, emb = self.extractor(x1, x2, x3)

        loss_m = self.loss_metrics(dist, alpha)
        loss_g = self.loss_generator(triplet, emb, dist, lmd, alpha)
        loss = loss_m + loss_g

        loss.backward()
        self.e_optim.step()
        self.g_optim.step()

        return loss_m, loss_g

    def loss_metrics(self, dist, alpha):
        self.e_optim.zero_grad()

        dist_a, dist_b = dist

        c = dist_a - dist_b + alpha
        loss = torch.clamp(c, min=0).mean()

        return loss

    def loss_generator(self, x, emb, dist, lmd, alpha):
        self.g_optim.zero_grad()

        x1, x2, x3 = x
        emb1, emb2, emb3 = emb
        dist_a, dist_b = dist
        lmd1, lmd2 = lmd

        inputs = torch.cat((emb1, emb2, emb3), dim=1)
        x3_fake = self.generator(inputs)

        loss_hard = self.mse(x3_fake, x1)
        loss_reg = self.mse(x3_fake, x3)
        loss_adv = torch.clamp(dist_b - dist_a - alpha, min=0).mean()

        loss = loss_hard + lmd1 * loss_reg + lmd2 * loss_adv

        return loss
