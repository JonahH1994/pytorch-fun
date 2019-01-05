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

# Dropout layer change parameter
training = 0
cuda = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.hcu = cuda
        
        self.layer1 = nn.Sequential(nn.Conv2d(1,6,5,stride=1,bias=False),nn.ReLU())
        self.layer2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.sig = nn.Sigmoid()
        self.layer3 = nn.Sequential(nn.Conv2d(6,16,5,stride=1,bias=False),nn.ReLU())
        self.layer4 = nn.MaxPool2d(kernel_size=2,stride=2) 
        self.layer5 = nn.Sequential(nn.Linear(16*5*5,120,bias=False),nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(120,84,bias=False),nn.ReLU())
        self.layer7 = nn.Linear(84,10,bias=False)

        self.dropout = nn.Dropout2d(0.5)
        self.dp = nn.Dropout2d(0)

    def forward(self, x):

        if self.hcu:
            x = x.to(torch.device('cuda:0'))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sig(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5(x)
        if training == 1:
            x = self.dropout(x).view(-1,120)
        else:
            x = self.dp(x).view(-1,120)
        x = self.layer6(x)
        if training == 1:
            x = self.dropout(x)
        else:
            x = self.dp(x)
        x = self.layer7(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Reshape(nn.Module):
    def __init__(self, args,cud):
        super(Reshape, self).__init__()
        self.shape = args
        if cud:
            self.cuda()

    def forward(self, x):
        return x.view(self.shape)

def train_batch(x, y, clf, opt, args):
    """Training step for one single batch (one forward, one backward, and one update)
    Args:
        [x]     FloatTensor (B, C, W, H), Images
        [y]     LongTensor  (B, 1), labels
        [clf]   Classifier model
        [opt]   Optimizer that updates the weights for the classifier.
        [args]  Commandline arguments.
    Rets:
        [loss]      (float) Loss of the batch (before update).
        [ncorrect]  (float) Number of correct examples in the batch (before update).
    """

    training = 1

    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    cpu = torch.device('cpu')

    clf.zero_grad()

    sofmax = nn.LogSoftmax(dim=1).to(device)

    output = clf.forward(x)
    LOSS = nn.CrossEntropyLoss()
    loss = LOSS(sofmax(output),y.to(device))
    loss.backward()

    pred = np.argmax(output.to(cpu).detach().numpy(),axis=1)
    ncorrect = np.sum((np.round(pred,decimals=2)==np.round(y.to(cpu).data.numpy(),decimals=2)).astype(int))

    opt.step()

    return loss, ncorrect


def evaluate(clf, loader, args):
    """Evaluate the classfier on the dataset loaded by the [loader]
    Args:
        [clf]       Model to be evaluated.
        [loader]    Dataloader, which will provide the test-set.
        [args]      Commandline arguments.
    Rets:
        Dictionary with following structure: {
            'loss'  : test loss (averaged across samples),
            'acc'   : test accuracy
        }
    """

    training = 0

    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    cpu = torch.device('cpu')
    
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss = 0
    acc = 0
    count = 0
    LS = nn.CrossEntropyLoss()
    for x,y in loader:
        output = clf(x)
        l = LS(output,y.to(device))
        pred = np.argmax(output.to(cpu).detach().data.numpy(),axis=1)
        results = (np.round(pred,decimals=1)==np.round(y.to(cpu).data.numpy(),decimals=1)).astype(int)
        c = np.sum(results)
        loss = loss + l
        acc = acc + c/float(x.size(0))
        count = count + 1

    loss = loss/count
    acc = acc/count
    rets = {'loss': loss, 'acc':acc}

    return rets


def resume_model(filename, args):
    """Resume the training (both model and optimizer) with the checkpoint.
    Args:
        [filename]  Str, file name of the checkpoint.
        [args]      Commandline arguments.
    Rets:
        [clf]   CNN with weights loaded with the pretrained weights from checkpoint [filename]
        [opt]   Optimizer with parameters resumed from the checkpoint [filename]
    """

    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dict_p = torch.load(filename)
    pretrained_dict = dict_p['clf']
    
    for key, value in pretrained_dict.items():
        #if 'weight' in key:
        print(key)
    clf = CNN().to(device)
    model_dict = clf.state_dict()

    weights = []

    diction = pretrained_dict
    for key, value in diction.items():
        if 'weight' in key:
            weights.append(key)

    count = 0
    max_count = len(weights)
    for key, value in model_dict.items():
        if 'weight' in key:
            if max_count-1 < count:
                print("More weights in pretrained dictionary than in architecture.")
            else:
                sz = model_dict[key].size()
                model_dict[key] = diction[weights[count]].view(sz).to(device)
                count +=1

    if count < max_count:
        print("Model has more weights than pretrained architecture.")
    elif count > max_count-1:
        pass
    else:
        count = 0
        for key, value in model_dict.items():
            if 'weight' in key:
                print(key,weights[count])
                count += 1

    clf.load_state_dict(model_dict)

    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    opt.load_state_dict(dict_p['opt'])

    return clf, opt

if __name__ == "__main__":
    args = utils.get_args()
    loaders = utils.get_loader(args)
    writer = SummaryWriter()
    cuda = args.cuda

    clf = CNN()
    if args.cuda:
        clf = clf.cuda()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    if args.resume is not None:
        clf, opt = resume_model(args.resume, args)

    step = 0
    for epoch in range(10):
        print("Epoch:%d"%(epoch+1))
        for x, y in loaders['train']:
            l, c = train_batch(x, y, clf, opt, args)
            acc = c/float(x.size(0))
            step += 1
            writer.add_scalar('Loss', l.item()) 
            writer.add_scalar('Number of Correct Classifications', c) 
            writer.add_image('Images', x)
            if step % 250 == 0:
                print("Step:%d\tLoss:%2.5f\tAcc:%2.5f"%(step, l, acc))

        print("Evaluating testing error:")
        res = evaluate(clf, loaders['test'], args)
        print(res)

    utils.save_checkpoint('clf.pth.tar', **{
        'clf' : clf.state_dict(),
        'opt' : opt.state_dict()
    })