# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg11
from torchvision import transforms

from daml.datasets import MNISTTripletDataset
from daml.updater import Updater
from daml.networks import FeatureExtractor, Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lmd1", type=float, default=1)
    parser.add_argument("--lmd2", type=float, default=1)
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    train_set = MNISTTripletDataset(
        root=args.root,
        n_samples=args.n_samples,
        train=(args.mode == "train"),
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    vgg = vgg11(pretrained=True)
    extractor = FeatureExtractor(
        nn.Sequential(*list(vgg.children())[:-1])).to(device)
    generator = Generator().to(device)

    e_optim = optim.Adam(extractor.parameters())
    g_optim = optim.Adam(generator.parameters())

    updater_dict = {
        "models": (extractor, generator),
        "optimizers": (e_optim, g_optim),
        "alpha": args.alpha,
        "lmd": (args.lmd1, args.lmd2),
    }
    updater = Updater(**updater_dict)

    for epoch in range(args.n_epochs):
        for idx, triplet in enumerate(train_loader):
            x1, x2, x3 = triplet
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

            loss_m, loss_g = updater.update((x1, x2, x3))
            print("metrics loss: ", loss_m, "generator loss: ", loss_g)


if __name__ == "__main__":
    main()
