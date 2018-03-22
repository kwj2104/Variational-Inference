import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import argparse
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--latdim", type=int, default=20)
    parser.add_argument("--layersize", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1)

    parser.add_argument("--optim", choices=["Adadelta", "Adam", "SGD"], default="SGD")

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()


def load_mnist():
    
    train_dataset = datasets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False, 
                               transform=transforms.ToTensor())
    
    
    
    # Why not take x > .5 etc?
    # Treat the greyscale values as probabilities and sample to get binary data
#    torch.manual_seed(3435)
#    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
#    train_label = torch.LongTensor([d[1] for d in train_dataset])
#    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
#    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
#    train_img = torch.stack([d[0] for d in train_dataset])
#    train_label = torch.LongTensor([d[1] for d in train_dataset])
#    test_img = torch.stack([d[0] for d in test_dataset])
#    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
    # Split training and val set
#    val_img = train_img[-10000:].clone()
#    val_label = train_label[-10000:].clone()
#    train_img = train_img[:10000]
#    train_label = train_label[:10000]
    
#    train = torch.utils.data.TensorDataset(train_img, train_label)
#    val = torch.utils.data.TensorDataset(val_img, val_label)
#    test = torch.utils.data.TensorDataset(test_img, test_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsize, shuffle=True)
#    val_loader = torch.utils.data.DataLoader(val, batch_size=args.bsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsize, shuffle=True)

    return train_loader, test_loader
    #return train_loader, val_loader, test_loader

# Compute the variational parameters for q
class NormalVAE(nn.Module):
    def __init__(self):
        super(NormalVAE, self).__init__()
        self.linear1_e = nn.Linear(784, args.layersize)
        self.linear2_e = nn.Linear(args.layersize, args.latdim)
        self.linear3_e = nn.Linear(args.layersize, args.latdim)
        self.linear1_d = nn.Linear(args.latdim, args.layersize)
        self.linear2_d = nn.Linear(args.layersize, 784)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        h = self.lrelu(self.linear1_e(x.view(-1, 784)))
        return self.linear2_e(h), self.linear3_e(h)
        
    def decode(self, z):
        out = self.linear2_d(self.relu(self.linear1_d(z)))      
        return self.sigmoid(out)
    
    def forward(self, x):
        
        #Encode
        mu, logvar = self.encode(x)
        
        #Sample
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())
        z_sample = q_normal.rsample()
        
        #Decode
        return self.decode(z_sample), q_normal

p = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                        V(torch.ones(args.bsize, args.latdim)))

seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                        V(torch.ones(args.bsize, args.latdim)))

def train(train_loader, model, loss_func):
    model.train()

    total_loss = 0
    total_kl = 0
    total = 0
    alpha = args.alpha
    for t in train_loader:
        img, label = t
        batch_size = img.size()[0]
        if batch_size == args.bsize:
            
            # Standard setup. 
            model.zero_grad()
            x = img.view(args.bsize, -1)
    
            # Run VAE. 
            out, q = model(x)
            kl = kl_divergence(q, p).sum()
            
#            out, logvar, mu = model(x)
#            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            #rec_loss = F.binary_cross_entropy(out, x)
            rec_loss = loss_func(out, x)
            
            loss = rec_loss + alpha * kl 
            loss = loss / batch_size
    
            # record keeping.
            total_loss += loss_func(out, x).data / batch_size
            total_kl += kl.data / batch_size
            total += 1
            loss.backward()
            optim.step()
        
    return total_loss / total, total_kl / total

def val(val_loader, model, loss_func):
    model.eval()

    total_loss = 0
    total_kl = 0
    total = 0
    alpha = args.alpha
    for t in val_loader:
        img, label = t
        batch_size = img.size()[0]
        if batch_size == args.bsize:
        
            x = img.view(args.bsize, -1)
    
            # Run VAE. 
            out, q = model(x)
            kl = kl_divergence(q, p).sum()
            
            rec_loss = loss_func(out, x)
            
            loss = rec_loss + alpha * kl 
            loss = loss / args.bsize
    
            # record keeping.
            total_loss += loss_func(out, x).data / args.bsize
            total_kl += kl.data / args.bsize
            total += 1
            
    return total_loss / total, total_kl / total



if __name__ == "__main__":
    train_loader, test_loader = load_mnist()
    
    model = NormalVAE()
    
    loss_func = nn.BCELoss(size_average=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    #Train
    i = 0
    print("Training..")
    for epoch in tqdm(range(args.epoch)):
        rec_loss, kl_loss = train(train_loader, model, loss_func)
        
    #Validate
        print("Testing..")
        rec_loss, kl_loss = val(test_loader, model, loss_func)
        print("Epoch: {} Reconstruction Loss: {} KL Loss: {}".format(i, rec_loss, kl_loss))
    
    #Save model
    torch.save(model.state_dict(), 'normalvae.pth')
    
    #Sample some new pics
    seed = seed_distribution.sample()
    fake = model.decode(seed).view(args.bsize, 1, 28, 28)
    vutils.save_image(fake.data,
            'vae_samples.png',
            normalize=True)
    
#    seed = seed_distribution.sample()
#    x = decoder(seed).view(args.bsize, 1, 28, 28)
#    for i in range(3):
#        fig, ax = plt.subplots()
#        ax.matshow(x.data[i][0], cmap=plt.cm.Blues)
        
    
        
    
    
    
    
    
    