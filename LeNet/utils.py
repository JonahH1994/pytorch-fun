import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from random import choice


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist only.')
    parser.add_argument('--dataroot', type=str, default='.', help='path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='File to resume')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='size of the latent z vector')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=256)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--e_ksize', type=int, default=4)
    parser.add_argument('--g_ksize', type=int, default=4)
    parser.add_argument('--d_ksize', type=int, default=4)
    parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--recon_l2_weight', type=float, default=0.5,
            help='Weight for L2 loss in the reconstruction of autoencoder.')
    parser.add_argument('--recon_l1_weight', type=float, default=0.5,
            help='Weight for L1 loss in the reconstruction of autoencoder.')
    parser.add_argument('--lr_enc', type=float, default=0.001,
            help='learning rate for encoder, default=0.001')
    parser.add_argument('--lr_dec', type=float, default=0.001,
            help='learning rate for decoder, default=0.001')
    parser.add_argument('--lr_g', type=float, default=0.0002,
            help='learning rate for G, default=0.0002')
    parser.add_argument('--lr_d', type=float, default=0.0002,
            help='learning rate for D, default=0.0002')
    parser.add_argument('--beta_1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--seed', default=100, type=int, help='Random seed.')
    args = parser.parse_args()

    return args


class ArithmeticsDataset(Dataset):
    def __init__(self, root_dir, length=50000):
        self.n = length

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        mnist = torchvision.datasets.MNIST(
            root=root_dir, train=True, transform=transform, download=True)

        data_dict = {} # label (int) -> image
        for l in range(10):
            data_dict[l] = []

        for x,y in mnist:
            data_dict[int(y)].append(x)

        self.data = data_dict

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        x = choice(range(5)) * 2
        y = 9 - x
        w = choice(range(5)) * 2 + 1
        z = 9 - w
        x_img = choice(self.data[x])
        y_img = choice(self.data[y])
        z_img = choice(self.data[z])
        w_img = choice(self.data[w])
        row1 = torch.cat([x_img, y_img], dim=2)
        row2 = torch.cat([z_img, w_img], dim=2)
        sample = torch.cat([row1, row2], dim=1)
        return sample, x


def save_checkpoint(filename, **states):
    torch.save(states, filename)


def get_loader(args):
    if args.dataset == 'mnist':
       # Image processing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.MNIST(
            root=args.dataroot, train=True, transform=transform, download=True)
        testset = torchvision.datasets.MNIST(
            root=args.dataroot, train=False, transform=transform, download=True)

    elif args.dataset == 'arith':
        dataset = ArithmeticsDataset(args.dataroot)
        testset = ArithmeticsDataset(args.dataroot)

    else:
        raise Exception("Invalid dataset:%s"%args.dataset)

    # Data loader
    data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
    )

    return {
        'train':data_loader,
        'test' :test_loader
    }


