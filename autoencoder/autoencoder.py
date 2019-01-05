#!/usr/bin/python3.5
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from random import choice
from . import utils


def recon_loss(g_out, labels, args):
    """
    Args:
        [g_out]     FloatTensor, (B, C, W, H), Output of the generator.
        [labels]    FloatTensor, (B, C, W, H), Ground truth images.
    Rets:
        Reconstruction loss with both L1 and L2.
    """
    lamda_1 = args.recon_l1_weight
    lamda_2 = args.recon_l2_weight

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    if args.cuda:
        device = torch.device('cuda:0')
        l1_loss.to(device)
        l2_loss.to(device)

        g_out = g_out.to(device)
        labels = labels.to(device)

    el1 = l1_loss(g_out,labels)
    el2 = l2_loss(g_out,labels)

    elem1 = torch.mul(el1,lamda_1)
    elem2 = torch.mul(el2,lamda_2)
    loss = torch.add(elem1, elem2)

    return loss


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        #raise NotImplementedError()
        self.num_filters = args.nef
        self.kernel_size = args.e_ksize
        self.channels = args.nc
        out_channel = args.nc
        self.hcu = args.cuda

        self.conv1 = nn.Sequential(nn.Conv2d(self.channels,self.num_filters,self.kernel_size,stride=2,padding=1,bias=False),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_filters,self.num_filters*2,self.kernel_size,stride=2,padding=1,bias=False),nn.BatchNorm2d(self.num_filters*2),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.num_filters*2,self.num_filters*4,self.kernel_size,stride=2,padding=1,bias=False),nn.BatchNorm2d(self.num_filters*4),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(self.num_filters*4,self.num_filters*8,self.kernel_size,stride=2,padding=1,bias=False),nn.BatchNorm2d(self.num_filters*8),nn.ReLU())
        self.FC = nn.Sequential(Reshape((-1,self.num_filters*8*4),args.cuda),nn.Linear(self.num_filters*8*4,args.nz,bias=False))

    def forward(self, x):

        if self.hcu:
            x = x.to(torch.device('cuda:0'))
        
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.FC(res)

        return res

    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'autoencoder.pth.tar' file below.)
        Usage:
            enet = Encoder(args)
            enet.load_model('autoencoder.pth.tar')
            # Here [enet] should be loaded with weights from file 'autoencoder.pth.tar'
        """
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()

        weights = []

        diction = pretrained_dict["encoder"]
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

class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.num_filters = args.ndf
        self.kernel_size = args.g_ksize
        self.channels = args.nz
        out_channel = args.nc
        self.hcu = args.cuda 

        self.relu = nn.ReLU()
        self.fc = nn.Linear(args.nz,args.ngf*4*4*4,bias=False)
        self.bn1 = nn.BatchNorm1d(args.ngf*4*4*4)
        self.reshape = nn.Sequential(Reshape((-1,args.ngf*4,4,4),args.cuda),
                                    nn.ReLU())
        self.conv1 = nn.ConvTranspose2d(args.ngf*4,args.ngf*2,args.g_ksize,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(args.ngf*2)
        self.conv2 = nn.ConvTranspose2d(args.ngf*2,args.ngf,args.g_ksize,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(args.ngf)
        self.conv3 = nn.ConvTranspose2d(args.ngf,args.nc,args.g_ksize,stride=2,padding=1,bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z, c=None):

        if self.hcu:
            z = z.to(torch.device('cuda:0'))

        z = self.fc(z)
        z = self.bn1(z)
        z = self.reshape(z)
        z = self.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.conv3(z)
        z = self.tanh(z)

        return z

    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'autoencoder.pth.tar' file below.)
        Usage:
            dnet = Decoder(args)
            dnet.load_model('autoencoder.pth.tar')
            # Here [dnet] should be loaded with weights from file 'autoencoder.pth.tar'
        """
        #raise NotImplementedError()
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()

        weights = []

        diction = pretrained_dict["decoder"]
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

class Reshape(nn.Module):
    def __init__(self, args,cud):
        super(Reshape, self).__init__()
        self.shape = args
        if cud:
            self.cuda()

    def forward(self, x):
        return x.view(self.shape)



def train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=None):
    """Train the AutoEncoder for one iteration (i.e. forward, backward, and update
       weights for one batch of data)
    Args:
        [input_data]    Input tensors tuple from the data loader.
        [encoder]       Encoder module.
        [decoder]       Decoder module.
        [enc_opt]       Optimizer to update encoder's weights.
        [dec_opt]       Optimizer to update decoder's weights.
        [args]          Commandline arguments.
        [writer]        Tensorboard writer (optional)
    Rets:
        [loss]  (float) Reconstruction loss of the batch (before the update).
    """
    
    encoder.zero_grad()
    decoder.zero_grad()

    input_data = input_data[0]

    encoder_output = encoder.forward(input_data)
    decoder_output = decoder.forward(encoder_output)
    loss = recon_loss(decoder_output,input_data,args)
    loss.backward()
    enc_opt.step()
    dec_opt.step()

    writer.add_scalar('Loss', loss.item()) 
    writer.add_image('Real Image', input_data)
    writer.add_image('Decoded Image', decoder_output)

    del decoder_output
    del encoder_output

    return loss


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
        if args.cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        b = args.batch_size

        iterations = int(n/b)
        output = torch.ones((), requires_grad=False)
        output = output.new_empty((n,args.nc,32,32))

        output = output.to(device)
        model = model.to(device)

        for i in range(iterations):
            #vals = sampler()
            output[i*b:(i+1)*b,:,:,:] = model(sampler().to(device))

        return output.cpu().detach().numpy()


if __name__ == "__main__":
    args = utils.get_args()
    loader = utils.get_loader(args)['train']
    writer = SummaryWriter()

    decoder = Decoder(args)
    encoder = Encoder(args)
    if args.cuda:
        decoder = decoder.cuda()
        encoder = encoder.cuda()
        print("Cuda enabled..")

    if args.encoder:
        try:
            encoder.load_model(args.encoder)
        except:
            print('Unable to open the given filename, training Encoder from scratch.')

    if args.decoder:
        try:
            decoder.load_model(args.decoder)
        except:
            print('Unable to open the given filename, training Decoder from scratch.')

    dec_opt = torch.optim.Adam(
            decoder.parameters(), lr=args.lr_dec, betas=(args.beta_1, args.beta_2))
    enc_opt = torch.optim.Adam(
            encoder.parameters(), lr=args.lr_enc, betas=(args.beta_1, args.beta_2))

    step = 0
    for epoch in range(args.nepoch):
        for input_data in loader:
            l = train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=writer)
            step += 1
            if step % 50 == 0:
                print("Step:%d\tLoss:%2.5f"%(step, l))
                writer.export_scalars_to_json("./all_scalars.json")
    
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    utils.save_checkpoint('autoencoder1.pth.tar', **{
        'decoder' : decoder.state_dict(),
        'encoder' : encoder.state_dict(),
        'dec_opt' : dec_opt.state_dict(),
        'enc_opt' : enc_opt.state_dict()
    })

    decoder.load_model('autoencoder1.pth.tar')
    encoder.load_model('autoencoder1.pth.tar')


    def get_z():
        z = torch.rand(args.batch_size, args.nz)
        if args.cuda:
            z = z.cuda(async=True)
        return z

    gen_img = sample(decoder, 60000, get_z, args)
    np.save('autoencoder_out.npy', gen_img)