# -*- coding: utf-8 -*-
import torch


class KMeans(object):
    def __init__(self, dist_function, n_clusters=10, max_iter=300):
        self.dist_function = dist_function
        self.n_clusters = n_clusters
        self.max_iter = 300

        self.C = None
        self.clusters = None

    def fit(self, x):
        self.C = torch.randn(self.n_clusters, x.size(1))
        self.clusters = torch.randn(x.size(0))

        C_old = self.C.clone()

        err = self.dist_function(self.C, C_old)

        it = 0
        while err > 0 or it > self.max_iter:
            for i in range(len(x)):
                dist = self.dist_function(x[i], self.C)
                cluster = torch.argmin(dist)
                self.clusters[i] = cluster

            C_old = self.C.clone()
            for i in range(self.n_clusters):
                points = [
                    x[j] for j in range(x.size(0)) if self.clusters[j] == i]
                self.C[i] = points.mean(dim=0)

            it += 1

            err = self.dist_function(self.C, C_old)
