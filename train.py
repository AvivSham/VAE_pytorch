"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms

import numpy as np
import VAE
import pickle
import matplotlib.pyplot as plt
import os

N = 100


def train(vae : VAE.Model, trainloader, optimizer, epoch, device):
    vae.train()  # set to training mode
    batch_loss = 0
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(inputs)
        loss = vae.loss(inputs, x_recon, mu=mu, logvar=logvar)
        loss.backward()
        batch_loss += loss.item()
        optimizer.step()
    return batch_loss, mu, logvar


def test(vae: VAE.Model, testloader, device: torch.device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        batch_loss = 0
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            x_recon, mu, logvar = vae(inputs)
            loss = vae.loss(inputs, x_recon, mu=mu, logvar=logvar)
            batch_loss += loss.item()
        return batch_loss



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = VAE.Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    elbo_train, elbo_val = [], []
    for epoch in range(args.epochs):
        train_loss, mu, logvar = train(vae=vae, trainloader=trainloader, optimizer=optimizer,
                                       epoch=epoch, device=device)
        elbo_train.append(train_loss / len(trainloader.dataset))
        elbo_val.append(test(vae=vae, testloader=testloader,device=device) / len(testloader.dataset))
    vae.sample(args.sample_size)
    fig, ax = plt.subplots()
    ax.plot(elbo_train)
    ax.plot(elbo_val)
    ax.set_title("Train/Test ELBO")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.legend(["train ELBO","test ELBO"])
    # ax.legend(["train loss"])
    plt.savefig(os.path.join(os.getcwd(),"loss",f"{args.dataset}_loss.png"))
    print("running done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=1)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)

