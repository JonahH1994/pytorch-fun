import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from random import choice
from . import utils
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.hcu = args.cuda

        relu = nn.LeakyReLU(negative_slope=0.2)
        
        layer1 = nn.Conv2d(args.nc,args.ndf,4,stride=2,padding=1,bias=False)
        layer2 = nn.Conv2d(args.ndf,args.ndf*2,4,stride=2,padding=1,bias=False)
        layer3 = nn.Conv2d(args.ndf*2,args.ndf*4,4,stride=2,padding=1,bias=False)
        layer4 = nn.Conv2d(args.ndf*4,1,4,stride=1,padding=0,bias=False)

        self.disc = nn.Sequential(layer1,relu,layer2,nn.BatchNorm2d(args.ndf*2),relu,
            layer3,nn.BatchNorm2d(args.ndf*4),relu,layer4,nn.Sigmoid())

    def forward(self, x):

        if self.hcu:
            x = x.to(torch.device('cuda:0'))
        
        return self.disc(x)


    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """

        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()

        weights = []

        diction = pretrained_dict["dnet"]
        for key, value in diction.items():
            if 'weight' in key:
                print(key)

class Reshape(nn.Module):
    def __init__(self, args,cud):
        super(Reshape, self).__init__()
        self.shape = args
        if cud:
            self.cuda()

    def forward(self, x):
        return x.view(self.shape)


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.hcu = args.cuda
        self.gen = nn.Sequential(nn.Linear(args.nz,args.ngf*4*4*4,bias=False),
                                 nn.BatchNorm1d(args.ngf*4*4*4),
                                 Reshape((-1,args.ngf*4,4,4),args.cuda),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(args.ngf*4,args.ngf*2,4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(args.ngf*2),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(args.ngf*2,args.ngf,4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(args.ngf),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(args.ngf,args.nc,4,stride=2,padding=1,bias=False),
                                 nn.Tanh())

    def forward(self, z, c=None):

        if self.hcu:
            z = z.to(torch.device('cuda:0'))

        return self.gen(z)

    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """

        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()

        weights = []

        diction = pretrained_dict["gnet"]
        for key, value in diction.items():
            if 'weight' in key:
                weights.append(key)

        count = 0
        max_count = len(weights)
        for key, value in model_dict.items():
            if 'weight' in key:
                if max_count < count:
                    print("More weights in pretrained dictionary than in architecture.")
                else:
                    model_dict[key] = diction[weights[count]]
                    count +=1

        if count < max_count:
            print("Model has more weights than pretrained architecture.")
        else:
            count = 0
            for key, value in model_dict.items():
                if 'weight' in key:
                    print(key,weights[count])
                    count += 1

        self.load_state_dict(model_dict)


def d_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Discriminator.
    """

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dreal = dreal.view(-1)
    dfake = dfake.view(-1)
    sz = dreal.size()
    rl = torch.ones(sz,device=device)
    fl = torch.zeros(sz,device=device)

    l1 = nn.BCELoss().to(device)
    m = nn.Sigmoid()

    loss1 = l1(dreal,rl)
    loss1.backward()
    loss2 = l1(dfake,fl)
    loss2.backward()
    loss = torch.add(loss1,loss2)

    return loss

def g_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Generator.
    """

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dfake = dfake.view(-1)
    sz = dfake.size()
    rl = torch.ones(sz,device=device)
    l1 = nn.BCELoss().to(device)
    loss = l1(dfake,rl)

    return loss


def train_batch(input_data, g_net, d_net, g_opt, d_opt, sampler, args, writer=None):
    """Train the GAN for one batch iteration.
    Args:
        [input_data]    Input tensors (tuple). Should contain the images and the labels.
        [g_net]         The generator.
        [d_net]         The discriminator.
        [g_opt]         Optimizer that updates [g_net]'s parameters.
        [d_opt]         Optimizer that updates [d_net]'s parameters.
        [sampler]       Function that could output the noise vector for training.
        [args]          Commandline arguments.
        [writer]        Tensorboard writer.
    Rets:
        [L_d]   (float) Discriminator loss (before discriminator's update step).
        [L_g]   (float) Generator loss (before generator's update step)
    """

    g_net.zero_grad()

    img = input_data[0]

    B= img.size(0)

    real_disc_output = d_net.forward(img)
    gen_output = torch.tensor(sample(g_net,B,sampler,args))
    fake_disc_output = d_net.forward(gen_output)
    loss_gen = g_loss(real_disc_output,fake_disc_output)

    loss_gen.backward()

    g_opt.step()

    d_net.zero_grad()
    gen_output_disc = torch.tensor(sample(g_net,B,sampler,args))
    fake_disc_output_disc = d_net.forward(gen_output_disc)
    loss_disc = d_loss(real_disc_output,fake_disc_output_disc)

    ### !!!!!!!!! Discriminator back propogation happens in loss function !!!!!! ###

    d_opt.step()

    # This is information that is used by tensorboard for visualization purposes. 
    # If you do not plan to use then comment out here as well as in the main function.
    writer.add_scalar('Loss/D', loss_disc.item())
    writer.add_scalar('Loss/G', loss_gen.item()) 
    writer.add_scalar('D(x)',real_disc_output.data.mean())
    writer.add_scalar('D(G(z1))', fake_disc_output.data.mean())
    writer.add_scalar('D(G(z2))', fake_disc_output_disc.data.mean())
    writer.add_image('Real Image', img)
    writer.add_image('Fake Image', gen_output)

    return loss_disc, loss_gen

def sample(model, n, sampler, args):
    """ Sample [n] images from [model] using noise created by the sampler.
    Args:
        [model]     Generator model that takes noise and output images.
        [n]         Number of images to sample.
        [sampler]   [sampler()] will return a batch of noise.
    Rets:
        [imgs]      (B, C, W, H) Float, numpy array.
    """
    with torch.no_grad():
        b = args.batch_size

        iterations = int(n/b)
        if iterations*b < n:
            remainder = n - b*iterations
        output = torch.zeros((n,args.nc,32,32))

        if args.cuda:
            output = output.to(torch.device('cuda:0'))

        prevInd = 0

        for i in range(iterations):
            vals = model.forward(sampler())

            sz = vals.size(0)

            output[prevInd:prevInd+sz,:,:,:] = vals
            prevInd += sz

        if prevInd != n:
            vals = model.forward(sampler())
            sz = vals.size(0)

            output[prevInd:n,:,:,:] = vals[:n-prevInd,:,:,:]

        if args.cuda:
            output = output.cpu()

        return output.detach().numpy()

if __name__ == "__main__":
    args = utils.get_args()
    loader = utils.get_loader(args)['train']
    writer = SummaryWriter(args.dataroot)

    d_net = Discriminator(args)
    g_net = Generator(args)

    if args.g_net:
        try: 
            g_net.load_model(args.g_net)
        except:
            print('Unable to open the given filename, training Generator from scratch.')

    if args.d_net:
        try:
            d_net.load_model(args.d_net)
        except:
            print('Unable to open the given filename, training Discriminator from scratch.')

    if args.cuda:
        d_net = d_net.cuda()
        g_net = g_net.cuda()

    d_opt = torch.optim.Adam(
            d_net.parameters(), lr=args.lr_d, betas=(args.beta_1, args.beta_2))
    g_opt = torch.optim.Adam(
            g_net.parameters(), lr=args.lr_g, betas=(args.beta_1, args.beta_2))

    def get_z():
        z = torch.rand(args.batch_size, args.nz)
        if args.cuda:
            z = z.cuda(async=True)
        return z

    step = 0
    for epoch in range(args.nepoch):
        for input_data in loader:
            l_d, l_g = train_batch(
                input_data, g_net, d_net, g_opt, d_opt, get_z, args, writer=writer)

            step += 1
            if step % 50 == 0:
                print("Epoch:%d\tStep:%d\tLossD:%2.5f\tLossG:%2.5f"%(epoch,step, l_d, l_g))
                writer.export_scalars_to_json("./all_scalars.json")

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    utils.save_checkpoint('dcgan.pth.tar', **{
        'gnet' : g_net.state_dict(),
        'dnet' : d_net.state_dict(),
        'gopt' : g_opt.state_dict(),
        'dopt' : d_opt.state_dict()
    })

    gen_img = sample(g_net, 60000, get_z, args)
    np.save('dcgan_out.npy', gen_img)