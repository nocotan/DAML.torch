# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class MNISTTripletDataset(data.Dataset):
    def __init__(self, root="./data", n_samples=5000,
                 transform=None, train=True):
        self.root = root
        self.n_samples = n_samples
        self.transform = transform
        self.train = train
        self.n_classes = 10

        self.triplets, self.imgs = self.make_triplets(root, n_samples, train)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        idx1, idx2, idx3 = self.triplets[index]
        img1, img2, img3 = self.imgs[idx1], self.imgs[idx2], self.imgs[idx3]

        img1 = Image.fromarray(img1.numpy(), mode="L")
        img2 = Image.fromarray(img2.numpy(), mode="L")
        img3 = Image.fromarray(img3.numpy(), mode="L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def make_triplets(self, root="./data", n_samples=5000, train=True):
        mnist = datasets.MNIST(
            root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )

        imgs = []
        for i in range(len(mnist)):
            imgs.append(mnist.__getitem__(i)[0].reshape(28, 28))

        if train:
            labels = mnist.train_labels
        else:
            labels = mnist.test_labels

        triplets = []
        for class_idx in range(self.n_classes):
            a = np.random.choice(np.where(labels == class_idx)[0],
                                 int(n_samples/self.n_classes),
                                 replace=True)
            b = np.random.choice(np.where(labels == class_idx)[0],
                                 int(n_samples/self.n_classes),
                                 replace=True)
            c = np.random.choice(np.where(labels != class_idx)[0],
                                 int(n_samples/self.n_classes),
                                 replace=True)

            for i in range(int(n_samples/self.n_classes)):
                triplets.append([int(a[i]), int(b[i]), int(c[i])])

        return triplets, imgs
