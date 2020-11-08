import math
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import random

from itertools import product
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms


import math
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.patches as patches

#import seaborn as sns; sns.set(color_codes=True)
from scipy.stats import kde

from itertools import product
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip","install",package])

install("POT")
#!pip install -U https://github.com/PythonOT/POT/archive/master.zip

import ot
import ot.plot

plt.style.use('bmh')
plt.rcParams["figure.figsize"] = (8,6)

def get_ot(x,y):
    if(len(x)!=len(y)): raise ValueError('Inputs do not have same length')
    
    n = len(x)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
    if(torch.is_tensor(x) and torch.is_tensor(y)):
        xc = x.detach().cpu().numpy()
        yc = y.detach().cpu().numpy()
    else:
        xc = x
        yc = y

    M=ot.dist(xc,yc, metric='euclidean')
    W1 = ot.emd2(a,b,M)
    return W1

def get_W1(data = None, fake_data = None, trains=None, traint = None, n_wb = 10, types = 'all'):
    if(types == 'all'):
        W1_total = get_ot(data,fake_data)

        sbatch = process_batch(trains)
        tbatch = process_batch(traint)

        W1_batch = get_ot(sbatch,tbatch)

        sbatches = sbatch
        tbatches = tbatch
        for i in range(n_wb-1):
            sbatches = torch.cat( (sbatches, process_batch(trains)) ,dim=0)
            tbatches = torch.cat( (tbatches, process_batch(traint)) ,dim=0)

        W1_morebat = get_ot(sbatches,tbatches)
        return (W1_batch, W1_morebat, W1_total)
    elif(types == 'batches'):
        sbatches = process_batch(trains)
        tbatches = process_batch(traint)
        for i in range(n_wb-1):
            sbatches = torch.cat( (sbatches, process_batch(trains)) ,dim=0)
            tbatches = torch.cat( (tbatches, process_batch(traint)) ,dim=0)

        W1_morebat = get_ot(sbatches,tbatches)
        return W1_morebat
    elif(types == 'batch'):
        sbatch = process_batch(trains)
        tbatch = process_batch(traint)
        
        W1_batch = get_ot(sbatch,tbatch)
        return W1_batch

def sep(data):
    if(torch.is_tensor(data)):
        d = data.cpu().detach().numpy()
    else:
        d = data
    x = [d[i][0] for i in range(len(d))]
    y = [d[i][1] for i in range(len(d))]
    return x,y

def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def compute_noise(batch_size, z_dim, noise = 'normal'):
    """ Compute random noise for input into the Generator G """
    if(noise == 'normal'):
        return to_cuda(torch.randn(batch_size, z_dim))
    elif(noise == 'unif'):
        return to_cuda(torch.rand(batch_size, z_dim))
    else:
        return to_cuda(noise(batch_size, z_dim))
    
    
    
def c_process_batch(iterator):
    """ Generate a process batch to be input into the Discriminator D """

    images, y = next(iterator)
    images = to_cuda(images.view(images.shape[0], -1))
    y = to_cuda(y.view(y.shape[0], -1))
    return images, y
    
def get_cdataloader(data, y, BATCH_SIZE=64, tt_split = 1, shuffle = True):
    """ Load data for binared MNIST """
    if(tt_split == 1):
        dset = to_cuda(torch.tensor(data).float())
        yset = to_cuda(torch.tensor(y).float())
        train = torch.utils.data.TensorDataset(dset, yset)
        train_loader = torch.utils.data.DataLoader(train,batch_size = BATCH_SIZE, shuffle = shuffle)
        return train_loader
    else:
        train_size = len(data)*tt_split
        train_dataset = to_cuda(torch.tensor(data[:int(np.ceil(train_size))]).float())
        train_yset = to_cuda(torch.tensor(y[:int(np.ceil(train_size))]).float())
        test_dataset = to_cuda(torch.tensor(data[int(np.ceil(train_size)):]).float())
        test_yset = to_cuda(torch.tensor(y[int(np.ceil(train_size)):]).float())
        
        train = torch.utils.data.TensorDataset(train_dataset, train_yset)
        test = torch.utils.data.TensorDataset(test_dataset, test_yset)
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = shuffle)
        test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = shuffle)
        return train_loader, test_loader

def get_dataloader(data, BATCH_SIZE=64, tt_split = 0.8, nosplit = False, shuffle = True, only_tt = False):
    """ Load data for binared MNIST """
    if(nosplit):
        dset = to_cuda(torch.tensor(data).float())
        train = torch.utils.data.TensorDataset(dset, torch.zeros(dset.shape[0]))
        train_loader = torch.utils.data.DataLoader(train,batch_size = BATCH_SIZE, shuffle = shuffle)
        return train_loader
        
    elif(only_tt):
        train_size = int(np.ceil(len(data)*tt_split))
        train_dataset = to_cuda(torch.tensor(data[:train_size]).float())
        test_dataset = to_cuda(torch.tensor(data[train_size:]).float())
        
        train = torch.utils.data.TensorDataset(train_dataset, to_cuda(torch.zeros(train_dataset.shape[0])))
        test = torch.utils.data.TensorDataset(test_dataset, to_cuda(torch.zeros(test_dataset.shape[0])))
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = shuffle)
        test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = shuffle)
        return train_loader, test_loader
    else:
        # split randomized data into train:split*0.9,    val: split*0.1,    test: 1-split
        train_size = len(data)*tt_split
        train_dataset = torch.tensor(data[:int(np.ceil(train_size*0.9))]).float()
        val_dataset = torch.tensor(data[int(np.ceil(train_size*0.9)):int(np.ceil(train_size))]).float()
        test_dataset = torch.tensor(data[int(np.ceil(train_size)):]).float()

        train_dataset = to_cuda(train_dataset)
        val_dataset = to_cuda(val_dataset)
        test_dataset = to_cuda(test_dataset)

        # Create data loaders
        train = torch.utils.data.TensorDataset(train_dataset, torch.zeros(train_dataset.shape[0]))
        val = torch.utils.data.TensorDataset(val_dataset, torch.zeros(val_dataset.shape[0]))
        test = torch.utils.data.TensorDataset(test_dataset, torch.zeros(test_dataset.shape[0]))

        train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=shuffle)
        val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=shuffle)
        test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=shuffle)

        return train_iter, val_iter, test_iter

# Style plots (decorator), never used
def plot_styling(func):
    def wrapper(*args, **kwargs):
        style = {'axes.titlesize': 24,
                 'axes.labelsize': 20,
                 'lines.linewidth': 3,
                 'lines.markersize': 10,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'panel.background': element_rect(fill="white"),
                 'panel.grid.major': element_line(colour="grey50"),
                 'panel.grid.minor': element_line(colour="grey50")
                }
        with plt.style.context((style)):
            ax = func(*args, **kwargs)      
    return wrapper

class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """
    def __init__(self, pg):
        super(Generator, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(pg[i], pg[i+1]) for i in range(len(pg)-2)])
        self.generate = nn.Linear(pg[-2], pg[-1])

    def forward(self, x):
        for l in self.linears:
            x = F.relu(l(x))
        x = self.generate(x)           # TODO : torch.sigmoid?
        return x
        

class Discriminator(nn.Module):
    """ Critic (not trained to classify). Input is an image (real or generated),
    output is the approximate Wasserstein Distance between z~P(G(z)) and real.
    """
    def __init__(self, pd):
        super(Discriminator, self).__init__()
        """   batchnorm
        self.linears = nn.ModuleList([nn.Linear(image_size, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        for i in range(n_layers):
            self.linears.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        """
        
        self.linears = nn.ModuleList([nn.Linear(pd[i], pd[i+1]) for i in range(len(pd)-2)])
        self.discriminate = nn.Linear(pd[-2], pd[-1])

    def forward(self, x):
        """  batchnorm
        for i, l in enumerate(self.linears):
            if i % 2 == 0:
                x = self.linears[i+1](F.relu(l(x)))
        x = self.discriminate(x)
        return x
        """

        for l in self.linears:
            x = F.relu(l(x))
        x = self.discriminate(x)   # TODO : torch.sigmoid?
        return x
    
    
class sigmoid_Discriminator(nn.Module):
    """ Critic (not trained to classify). Input is an image (real or generated),
    output is the approximate Wasserstein Distance between z~P(G(z)) and real.
    """
    def __init__(self, image_size, hidden_dim, n_layers, output_dim):
        super(sigmoid_Discriminator, self).__init__()
        """   batchnorm
        self.linears = nn.ModuleList([nn.Linear(image_size, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        for i in range(n_layers):
            self.linears.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        """
        self.linears = nn.ModuleList([nn.Linear(pd[i], pd[i+1]) for i in range(len(pd)-2)])
        self.discriminate = nn.Linear(pd[-2], pd[-1])

    def forward(self, x):
        """  batchnorm
        for i, l in enumerate(self.linears):
            if i % 2 == 0:
                x = self.linears[i+1](F.relu(l(x)))
        x = self.discriminate(x)
        return x
        
        """
        for l in self.linears:
            x = F.relu(l(x))
        x = torch.sigmoid(self.discriminate(x))  # TODO : torch.sigmoid?
        return x
    

class WGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, pg, pd):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(pg)
        self.D = Discriminator(pd)
        self.pg = pg
        self.pd = pd

        #self.shape = int(image_size ** 0.5)
        
class GAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, pg, pd):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(pg)
        self.D = sigmoid_Discriminator(pd)
        self.pg = pg
        self.pd = pd


        
class cGenerator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """
    def __init__(self, pg):
        super(cGenerator, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(pg[i], pg[i+1]) for i in range(len(pg)-2)])
        self.generate = nn.Linear(pg[-2], pg[-1])

    def forward(self, x,y):
        x = torch.cat((x,y),1)
        for l in self.linears:
            x = F.relu(l(x))
        x = self.generate(x)           # TODO : torch.sigmoid?
        return x
        

class cDiscriminator(nn.Module):
    """ Critic (not trained to classify). Input is an image (real or generated),
    output is the approximate Wasserstein Distance between z~P(G(z)) and real.
    """
    def __init__(self, pd):
        super(cDiscriminator, self).__init__()
        """   batchnorm
        self.linears = nn.ModuleList([nn.Linear(image_size, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        for i in range(n_layers):
            self.linears.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])
        """
        
        self.linears = nn.ModuleList([nn.Linear(pd[i], pd[i+1]) for i in range(len(pd)-2)])
        self.discriminate = nn.Linear(pd[-2], pd[-1])

    def forward(self, x,y):
        """  batchnorm
        for i, l in enumerate(self.linears):
            if i % 2 == 0:
                x = self.linears[i+1](F.relu(l(x)))
        x = self.discriminate(x)
        return x
        """

        x=torch.cat((x,y), 1)
        for l in self.linears:
            x = F.relu(l(x))
        x = self.discriminate(x)   # TODO : torch.sigmoid?
        return x
    
class cWGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, pg, pd):
        super().__init__()

        self.__dict__.update(locals())

        self.G = cGenerator(pg)
        self.D = cDiscriminator(pd)
        self.pg = pg
        self.pd = pd
        self.x_dim = pg[-1]
        self.y_dim = pd[0] - pg[-1]
        self.z_dim = pg[0]- self.y_dim

        
        
        
        
#conditional wgan trainer
class cWGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, trainset, testset = None, datas = None, datat = None, gantype = 'wgangp', noise = 'normal'):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__
        self.x_dim = model.x_dim
        self.y_dim = model.y_dim
        self.z_dim = model.z_dim
        self.noise = noise

        self.trains = trainset
        self.testset = testset
        self.iters = iter(trainset)
        if(testset is None):
            self.itertest = None
        else:
            self.itertest = iter(testset)
        
        self.datas = datas
        self.datat = datat

        self.Glosses = []
        self.Dlosses = []
        self.W1s = []
        self.W1test = []
        self.Wtotal = None

        self.num_epochs = 0
        self.gantype = gantype
        self.G_iter = 0
        self.D_iter = 0
        
    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        try:
            images, y = next(iterator)
        except StopIteration:
            if(iterator == self.iters):
                self.iters = iter(self.trains)
                images, y = next(self.iters)
            elif(iterator == self.itertest):
                self.itertest = iter(self.testset)
                images, y = next(self.itertest)
            else:
                print('Unknown Iterator')
            
        images = to_cuda(images.view(images.shape[0], -1))
        y = to_cuda(y.view(y.shape[0],-1))
        return images, y      #xy = torch.cat((images, y), 1)
        
    def get_gradients(self, distr, y, other_distr):
        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(distr, y) # D(z)
        DG_score = self.model.D(other_distr, y) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(distr.shape[0], 1).expand(distr.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        point_between = epsilon*distr + (1-epsilon)*other_distr
        D_interpolation = self.model.D(point_between, y)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=point_between,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]
        return gradients, DX_score, DG_score



    def train_D_step(self, distr, y, other_distr, LAMBDA=0.1):
        gradients, DX_score, DG_score = self.get_gradients(distr, y, other_distr)
        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss


    def W_estim(self, distr, y, other_distr):
        gradients, DX_score, DG_score = self.get_gradients(distr, y, other_distr)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score)
        return -D_loss
    
    def get_W1(self, n_wb = 10, types = 'batch', test = False):
        
        if(types == 'batches'):
            sbatches, ys = self.process_batch(self.iters)
            for i in range(n_wb-1):
                sb, y = self.process_batch(self.iters)
                sbatches = torch.cat( (sbatches, sb) ,dim=0)
                ys = torch.cat( (ys, y), dim=0)
            tbatches = self.generate_samples(ys, num_outputs = sbatches.shape[0])
            gens = torch.cat((tbatches,ys),dim=1)
            reals = torch.cat((sbatches,ys),dim=1)
            if(test):
                testbatches, ytest = self.process_batch(self.itertest)
                for i in range(n_wb-1):
                    sb, y = self.process_batch(self.itertest)
                    testbatches = torch.cat( (testbatches, sb) ,dim=0)
                    ytest = torch.cat( (ytest, y), dim=0)
                faketest = self.generate_samples(ytest, num_outputs = testbatches.shape[0])
                gens2 = torch.cat((faketest,ytest),dim=1)
                reals2 = torch.cat((testbatches,ytest),dim=1)
                return get_ot(reals,gens), get_ot(reals2,gens2)

            return get_ot(reals,gens)
        elif(types == 'batch'):
            sbatch, ys = self.process_batch(self.iters)
            tbatch = self.generate_samples(ys, num_outputs = sbatch.shape[0])
            gens = torch.cat((tbatch,ys),dim=1)
            reals = torch.cat((sbatches,ys),dim=1)
            if(test):
                testbatch,ytest = self.process_batch(self.itertest)
                faketest = self.generate_samples(ytest, num_outputs = sbatch.shape[0])
                gens2 = torch.cat((faketest,ytest),dim=1)
                reals2 = torch.cat((testbatch,ytest),dim=1)
                return get_ot(reals,gens), get_ot(reals2, gens2)
            return get_ot(reals,gens)            
            
            
    def train_estim(self, num_epochs = 1, penalty = 0.1, G_lr=1e-4, D_lr=1e-4, G_wd = 0, D_wd = 0, D_steps_standard=5, n_wb = 10, num_batches = 1, num_estims = 10, pot = True):
        """ Train a Wasserstein GAN
            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.
        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's RMProp optimizer
            D_lr: float, learning rate for discriminator's RMSProp optimizer
            D_steps: int, ratio for how often to train D compared to G
            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz
        """
        # Initialize optimizers
        bet = (0.5, 0.9)
        
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                        if p.requires_grad], lr=G_lr, weight_decay = G_wd,betas=bet)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr,weight_decay = D_wd, betas=bet)

        n_batches = len(self.trains)
        D_steps = D_steps_standard

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):
            #Train discriminator to almost convergence to approx W
            if( (self.G_iter <= 25) or (self.G_iter % 100 == 0) ):
                D_steps = 100
            else:               
                D_steps = D_steps_standard
                    
            self.model.train()
            G_losses, D_losses = [], []
            ep_iter = 0

            while(ep_iter < n_batches):
                D_step_loss = []

                for _ in range(D_steps):

                    # Reshape images
                    images, y = self.process_batch(self.iters)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D_GP(images, y, LAMBDA=penalty)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())
                    self.D_iter += 1
                    ep_iter += 1


                # We report D_loss in this way so that G_loss and D_loss have
                # the same number of entries.
                D_losses.append(np.mean(D_step_loss))
                
                '''
                # Visualize generator progress
                if self.viz:
                    if(self.G_iter < 200 and self.G_iter % 10 == 0):
                        self.viz_data(save = True)
                    elif(self.G_iter >200 and self.G_iter % 200 == 0):
                        self.viz_data(save=True)
                '''
                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the generator to (roughly) minimize the approximated
                # Wasserstein distance
                G_loss = self.train_G_W(images, y)
                
                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()
                self.G_iter += 1
                
            

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)
            if(pot):
                w1s = []
                w1t = []
                if(self.testset is not None):
                    for i in range(num_estims):
                        w1,w1test = self.get_W1(n_wb = num_batches, types = 'batches', test = True)
                        w1s.append(w1)
                        w1t.append(w1test)
                    self.W1s.append([np.mean(w1s),np.std(w1s)])
                    self.W1test.append([np.mean(w1t),np.std(w1t)])
                else:
                    w1s = [self.get_W1(n_wb = num_batches, types = 'batches') for i in range(num_estims)]
                    self.W1s.append([np.mean(w1s), np.std(w1s)])
            self.num_epochs += 1
    
    
    def train_D_GP(self, images, y, LAMBDA=0.1):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = compute_noise(images.shape[0], self.z_dim, noise = self.noise)
        G_output = self.model.G(noise, y)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images, y) # D(z)
        DG_score = self.model.D(G_output, y) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon*images + (1-epsilon)*G_output
        D_interpolation = self.model.D(G_interpolation, y)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss
    
    def train_D_LP(self, images, y, LAMBDA=0.1):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[max(0,∇ D(εx + (1 − εG(z))) - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = compute_noise(images.shape[0], self.z_dim, noise = self.noise)
        G_output = self.model.G(noise, y)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images, y) # D(z)
        DG_score = self.model.D(G_output, y) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon*images + (1-epsilon)*G_output
        D_interpolation = self.model.D(G_interpolation, y)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        zer = torch.zeros(gradients.norm(2,dim=1).shape[0])
        grad_penalty = LAMBDA * torch.mean((torch.max(zer, gradients.norm(2, dim=1) - 1))**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss
    

    def train_D_W(self, images, y):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))]
        """
        # Sample from the generator
        noise = compute_noise(images.shape[0], self.z_dim, noise = self.noise)
        G_output = self.model.G(noise, y)

        # Score real, generated images
        DX_score = self.model.D(images, y) # D(x), "real"
        DG_score = self.model.D(G_output, y) # D(G(x')), "fake"

        # Compute WGAN loss for D
        D_loss = -1 * (torch.mean(DX_score)) + torch.mean(DG_score)

        return D_loss
    
    


    def train_G_W(self, images, y):
        """ Run 1 step of training for generator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = compute_noise(images.shape[0], self.z_dim, noise = self.noise) # z
        G_output = self.model.G(noise, y) # G(z)
        DG_score = self.model.D(G_output, y) # D(G(z))

        # Compute WGAN loss for G
        G_loss = -1 * (torch.mean(DG_score))

        return G_loss
    
    
    def pretrain(self, G_init, G_optimizer):
        # Let G train for a few steps before beginning to jointly train G
        # and D because MM GANs have trouble learning very early on in training
        if G_init > 0:
            for _ in range(G_init):
                # Process a batch of images
                images, y = self.process_batch(self.iters)

                # Zero out gradients for G
                G_optimizer.zero_grad()

                # Pre-train G
                G_loss = self.train_G_W(images, y)

                # Backpropagate the generator network
                G_loss.backward()
                G_optimizer.step()

            print('G pre-trained for {0} training steps.'.format(G_init))
        else:
            print('G not pre-trained -- GAN unlikely to converge.')


    def clip_D_weights(self, clip):
        for parameter in self.model.D.parameters():
            parameter.data.clamp_(-clip, clip)
            
    
    def generate_samples(self, y, num_outputs=64):
        """ Visualize progress of generator learning
        careful! y must have lentgh num_outputs!
        """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = compute_noise(num_outputs, self.z_dim, noise = self.noise)

        # Transform noise to image
        images = self.model.G(noise, y)
        
        return images

        """
        # Reshape to square image size
        images = images.view(images.shape[0],
                             self.model.shape,
                             self.model.shape,
                             -1).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1


        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         %(epoch), nrow=grid_size)
        """

    def viz_loss(self, save = False):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.G_iter, len(self.Dlosses)),
                 self.Dlosses,
                 'b')
        print(self.num_epochs,self.G_iter, len(self.Dlosses))

        """
        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        """
        #plt.title(self.name)
        if save:
            plt.savefig('loss_%s_%d.png'%(self.gantype, self.G_iter), dpi = 150)
        plt.show()
        
    """    
    def viz_data(self, save = False, density=True):
        if(density == True):
            lower = (-1.3, -1.3)
            upper = (1.3, 1.3)
            nbins=300
            x,y = sep(self.generate_images(num_outputs = 1000))
            k = kde.gaussian_kde([x,y])
            xi, yi = np.mgrid[lower[0]:upper[1]:nbins*1j, lower[1]:upper[1]:nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            fig, ax = plt.subplots()

            # a (1-alpha)-confidence circle for 2d gaussian with cov = a * eyes has radius sqrt(-2 a ln(alpha)), here a = 3/400
            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues)
            for i in range(len(means)):
                circle = plt.Circle((means[i][0]/10, means[i][1]/10),           # (x,y)
                np.sqrt(-3 * np.log(0.05)/200), color='black',linewidth = 2, fill=False)
                ax.add_artist(circle)
            #plt.colorbar()
            
        else:
            # 2-dim points rowwise concatenated in array
            # Set style, figure size
            plt.style.use('ggplot')
            plt.rcParams["figure.figsize"] = (8,6)

            # Plot generated points in red
            gen = self.generate_samples()
            xg = [gen[i][0] for i in range(len(gen))]
            yg = [gen[i][1] for i in range(len(gen))]
            plt.plot(xg,
                     yg,
                     'ro', markersize = 3)


            # Plot real data points in green
            real = next(iter(train_iter))[0]
            xr = [real[i][0] for i in range(len(real))]
            yr = [real[i][1] for i in range(len(real))]
            plt.plot(xr,
                     yr,
                     'go', markersize = 3)

            # Add legend, title
            plt.legend(['generated', 'real'])
        
        #plt.title(self.name)
        if save:
            plt.savefig('data_%s_%d_%d.png'%(self.gantype,self.num_epochs, self.G_iter), dpi = 150)
        plt.show()
        """

    def save_model(self):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), 'model_%s_%d'%(self.gantype, self.G_iter))

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class WGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, trainset, traint = None, datas = None, datat = None, testset = None, viz=False, gantype = 'wgangp', noise = 'normal'):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__
        self.noise = noise

        self.trains = trainset
        self.iters = iter(trainset)
        self.traint = traint
        self.testset = testset
        if(traint is None):
            self.itert = None
        else:
            self.itert = iter(traint)
        
        if(testset is None):
            self.itertest = None
        else:
            self.itertest = iter(testset)
        self.datas = datas
        self.datat = datat

        self.Glosses = []
        self.Dlosses = []
        self.Wcs = []
        self.Wcns = []
        self.Wcms = []
        self.W1s = []
        self.W1test = []
        self.Wtotal = None

        self.viz = viz
        self.num_epochs = 0
        self.gantype = gantype
        self.G_iter = 0
        self.D_iter = 0
        
    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        try:
            images, _ = next(iterator)
        except StopIteration:
            if(iterator == self.iters):
                self.iters = iter(self.trains)
                images, _ = next(self.iters)
            elif(iterator == self.itert):
                self.itert = iter(self.traint)
                images, _ = next(self.itert)
            elif(iterator == self.itertest):
                self.itertest = iter(self.testset)
                images, _ = next(self.itertest)
            else:
                print('Unknown Iterator')
            
        images = to_cuda(images.view(images.shape[0], -1))
        return images
        
    def get_gradients(self, distr, other_distr):
        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(distr) # D(z)
        DG_score = self.model.D(other_distr) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(distr.shape[0], 1).expand(distr.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        point_between = epsilon*distr + (1-epsilon)*other_distr
        D_interpolation = self.model.D(point_between)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=point_between,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]
        return gradients, DX_score, DG_score



    def train_D_step(self, distr, other_distr, LAMBDA=0.1):
        gradients, DX_score, DG_score = self.get_gradients(distr,other_distr)
        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss


    def W_estim(self, distr, other_distr):
        gradients, DX_score, DG_score = self.get_gradients(distr,other_distr)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score)
        return -D_loss
    
    
    
    def get_W1(self, n_wb = 10, types = 'batch', test = False):
        
        if(types == 'batches'):
            if(self.traint is None):
                sbatches = self.process_batch(self.iters)
                for i in range(n_wb-1):
                    sb = self.process_batch(self.iters)
                    sbatches = torch.cat( (sbatches, sb) ,dim=0)
                tbatches = self.generate_samples(num_outputs = sbatches.shape[0])
                if(test):
                    testbatches = self.process_batch(self.itertest)
                    for i in range(n_wb-1):
                        sb = self.process_batch(self.itertest)
                        testbatches = torch.cat( (testbatches, sb) ,dim=0)
                    faketest = self.generate_samples(num_outputs = testbatches.shape[0])
                    return get_ot(sbatches,tbatches), get_ot(testbatches,faketest)
                
                return get_ot(sbatches,tbatches)
            else:
                print('W1 not implemented')
        
        elif(types == 'batch'):
            sbatch = self.process_batch(self.iters)
            if(self.traint is None):
                tbatch = self.generate_samples(num_outputs = sbatch.shape[0])
            if(test):
                testbatch = self.process_batch(self.itertest)
                faketest = self.generate_samples(num_outputs = sbatch.shape[0])
                return get_ot(testbatch,faketest), get_ot(testbatch, faketest)
            return get_ot(sbatch,tbatch), get_ot(reals,gens)    
    
    def get_W1_old(self, n_wb = 10, types = 'all'):
        if(types == 'all'):
            if(self.datat is None):
                datat = self.generate_samples(num_outputs = self.datas.shape[0])
                self.Wtotal = get_ot(self.datas, datat)
                sbatch = self.process_batch(self.iters)
                tbatch = datat[:sbatch.shape[0]]
                sbatches = sbatch
                
                for i in range(n_wb-1):
                    sbatches = torch.cat( (sbatches, self.process_batch(self.iters)) ,dim=0)
                tbatches = datat[:sbatches.shape[0]]
                    
            else:
                if(self.Wtotal is None):
                    self.Wtotal = get_ot(self.datas, self.datat)
                sbatch = self.process_batch(self.iters)
                tbatch = self.process_batch(self.itert)
                sbatches = sbatch
                tbatches = tbatch
                for i in range(n_wb-1):
                    sbatches = torch.cat( (sbatches, self.process_batch(self.iters)) ,dim=0)
                    tbatches = torch.cat( (tbatches, self.process_batch(self.itert)) ,dim=0)

            W1_batch = get_ot(sbatch,tbatch)
            W1_morebat = get_ot(sbatches,tbatches)
            return (W1_batch, W1_morebat, self.Wtotal)
        
        elif(types == 'batches'):
            if(self.traint is None):
                sbatches = self.process_batch(self.iters)
                for i in range(n_wb-1):
                    sbatches = torch.cat( (sbatches, self.process_batch(self.iters)) ,dim=0)
                tbatches = self.generate_samples(num_outputs = sbatches.shape[0])
                
            else:
                sbatches = self.process_batch(self.iters)
                tbatches = self.process_batch(self.itert)
                for i in range(n_wb-1):
                    sbatches = torch.cat( (sbatches, self.process_batch(self.iters)) ,dim=0)
                    tbatches = torch.cat( (tbatches, self.process_batch(self.itert)) ,dim=0)

            W1_morebat = get_ot(sbatches,tbatches)
            return W1_morebat
        
        elif(types == 'batch'):
            sbatch = self.process_batch(self.iters)
            if(self.traint is None):
                tbatch = self.generate_samples(num_outputs = sbatch.shape[0])
            else:
                tbatch = self.process_batch(self.itert)

            W1_batch = get_ot(sbatch,tbatch)
            return W1_batch
    
    def train_D(self, num_epochs = 1, lr = 1e-4, wd = 0, num_batches = 1, num_estims = 10, onlypot = True):
        bet = (0.5, 0.9)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=lr,betas=bet, weight_decay = wd)

        n_batches = len(self.trains)
        self.model.train()
        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):
            self.model.train()

            ep_iter = 0
            while(ep_iter < n_batches):
                D_step_loss = []
                # Reshape images
                distr = self.process_batch(self.iters)
                if(self.traint is None):
                    other_distr = self.generate_samples(num_outputs = distr.shape[0])
                else:
                    other_distr = self.process_batch(self.itert)

                # TRAINING D: Zero out gradients for D
                D_optimizer.zero_grad()

                # Train the discriminator to approximate the Wasserstein
                # distance between real, generated distributions
                D_loss = self.train_D_step(distr, other_distr)
                
                if(onlypot):
                    w1s = [self.get_W1(types = 'batch') for i in range(num_estims)]
                    self.W1s.append([np.mean(w1s), np.std(w1s)])
                else:
                    wcs = []
                    wcns = []
                    wcms = []
                    if(num_batches == 1):
                        w1s = [self.get_W1(types = 'batch') for i in range(num_estims)]
                        for i in range(num_estims):
                            distr = self.process_batch(self.iters)
                            if(self.traint is None):
                                other_distr = self.generate_samples(num_outputs = distr.shape[0])
                            else:
                                other_distr = self.process_batch(self.itert)
                            grad = self.get_gradients(distr,other_distr)[0]
                            normalize = torch.mean(grad.norm(2, dim=1)).item()
                            wc = self.W_estim(distr, other_distr).item()
                            wcs.append(wc)
                            wcns.append(wc/ normalize)
                            wcms.append(wc/ torch.max(grad).item())

                    else:
                        w1s = [self.get_W1(n_wb = num_batches, types = 'batches') for i in range(num_estims)]
                        for i in range(num_estims):
                            maxes = []
                            wc_temps = []
                            norms = []
                            for _ in range(num_batches):
                                distr = self.process_batch(self.iters)
                                if(self.traint is None):
                                    other_distr = self.generate_samples(num_outputs = distr.shape[0])
                                else:
                                    other_distr = self.process_batch(self.itert)
                                grad = self.get_gradients(distr,other_distr)[0]
                                norms.append( torch.mean(grad.norm(2, dim=1)).item() )
                                wc_temps.append( self.W_estim(distr, other_distr).item() )
                                maxes.append( torch.max(grad).item() )

                            wcs.append(np.mean(wc_temps))
                            wcns.append(np.mean(wc_temps)/ np.mean(norms))
                            wcms.append(np.mean(wc_temps)/ np.mean(maxes))


                    self.W1s.append([np.mean(w1s), np.std(w1s)])
                    self.Wcs.append([np.mean(wcs), np.std(wcs)])
                    self.Wcns.append([np.mean(wcns), np.std(wcns)])
                    self.Wcms.append([np.mean(wcms), np.std(wcms)])

                # Update parameters
                D_loss.backward()
                D_optimizer.step()

                # Log results, backpropagate the discriminator network
                self.Dlosses.append(D_loss.item())
                self.D_iter += 1
                ep_iter += 1
            self.num_epochs += 1
            
            
    def train_estim(self, num_epochs = 1, penalty = 0.1, G_lr=1e-4, D_lr=1e-4, G_wd = 0, D_wd = 0, D_steps_standard=5, n_wb = 10, num_batches = 1, num_estims = 10, pot = True):
        """ Train a Wasserstein GAN
            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.
        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's RMProp optimizer
            D_lr: float, learning rate for discriminator's RMSProp optimizer
            D_steps: int, ratio for how often to train D compared to G
            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz
        """
        # Initialize optimizers
        bet = (0.5, 0.9)
        
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                        if p.requires_grad], lr=G_lr, weight_decay = G_wd,betas=bet)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr,weight_decay = D_wd, betas=bet)

        n_batches = len(self.trains)
        D_steps = D_steps_standard

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):
            #Train discriminator to almost convergence to approx W
            if( (self.G_iter <= 25) or (self.G_iter % 100 == 0) ):
                D_steps = 100
            else:               
                D_steps = D_steps_standard
                    
            self.model.train()
            G_losses, D_losses = [], []
            ep_iter = 0

            while(ep_iter < n_batches):
                D_step_loss = []

                for _ in range(D_steps):

                    # Reshape images
                    images = self.process_batch(self.iters)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D_GP(images, LAMBDA = penalty)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())
                    self.D_iter += 1
                    ep_iter += 1


                # We report D_loss in this way so that G_loss and D_loss have
                # the same number of entries.
                D_losses.append(np.mean(D_step_loss))
                
                '''
                # Visualize generator progress
                if self.viz:
                    if(self.G_iter < 200 and self.G_iter % 10 == 0):
                        self.viz_data(save = True)
                    elif(self.G_iter >200 and self.G_iter % 200 == 0):
                        self.viz_data(save=True)
                '''
                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the generator to (roughly) minimize the approximated
                # Wasserstein distance
                G_loss = self.train_G_W(images)
                
                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()
                self.G_iter += 1
                
            

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)
            
            if(pot):
                w1s = []
                w1t = []
                if(self.testset is not None):
                    for i in range(num_estims):
                        w1,w1test = self.get_W1(n_wb = num_batches, types = 'batches', test = True)
                        w1s.append(w1)
                        w1t.append(w1test)
                    self.W1s.append([np.mean(w1s),np.std(w1s)])
                    self.W1test.append([np.mean(w1t),np.std(w1t)])
                else:
                    w1s = [self.get_W1(n_wb = num_batches, types = 'batches') for i in range(num_estims)]
                    self.W1s.append([np.mean(w1s), np.std(w1s)])
            
            self.num_epochs += 1
    

    
    def train(self, num_epochs, G_lr=5e-5, D_lr=5e-5, G_wd = 0, D_steps_standard=5, clip=0.01, G_init=5, G_per_D = 1):
        """ Train a Wasserstein GAN
            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.
        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's RMProp optimizer
            D_lr: float, learning rate for discriminator's RMSProp optimizer
            D_steps: int, ratio for how often to train D compared to G
            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz
        """
        # Initialize optimizers
        if(self.gantype in ['wgan', 'wgangp', 'wganlp', 'ls']):
            bet = (0.5,0.9)
        else:
            bet = (0.5, 0.9)
        
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                        if p.requires_grad], lr=G_lr, weight_decay = G_wd,betas=bet)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr,betas=bet)

        num_batches = len(self.trains)
        D_steps = D_steps_standard
        self.model.train()
        if(self.gantype in ['gan','nsgan']):
                self.pretrain(G_init, G_optimizer)

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):
            #Train discriminator to almost convergence to approx W
            if(self.gantype in ['wgangp', 'wgan','wganlp']):
                if( (self.G_iter <= 25) or (self.G_iter % 100 == 0) ):
                    D_steps = 100
                else:               
                    D_steps = D_steps_standard
            else:
                D_steps = D_steps_standard
                    

            self.model.train()
            G_losses, D_losses = [], []
            ep_iter = 0

            while(ep_iter < num_batches):

                if(self.G_iter % G_per_D == 0):
                    D_step_loss = []

                    for _ in range(D_steps):

                        # Reshape images
                        images = self.process_batch(self.iters)

                        # TRAINING D: Zero out gradients for D
                        D_optimizer.zero_grad()

                        # Train the discriminator to approximate the Wasserstein
                        # distance between real, generated distributions
                        if(self.gantype == 'wgangp'):
                            D_loss = self.train_D_GP(images)
                        elif(self.gantype == 'wganlp'):
                            D_loss = self.train_D_LP(images, LAMBDA = 1)
                        elif(self.gantype == 'wgan'):
                            D_loss = self.train_D_W(images)
                        elif(self.gantype in ['nsgan','gan']):
                                D_loss = self.train_D_vanilla(images)
                        elif(self.gantype == 'ls'):
                            D_loss = self.train_D_ls(images)
                        else:
                            print('Unknown gantype')
                            break


                        # Update parameters
                        D_loss.backward()
                        D_optimizer.step()

                        # Log results, backpropagate the discriminator network
                        D_step_loss.append(D_loss.item())
                        self.D_iter += 1
                        ep_iter += 1

                        if(self.gantype == 'wgan'):
                            # Clamp weights (crudely enforces K-Lipschitz)
                            self.clip_D_weights(clip)

                    # We report D_loss in this way so that G_loss and D_loss have
                    # the same number of entries.
                    D_losses.append(np.mean(D_step_loss))
                else:
                    images = self.process_batch(self.iters)

                '''
                # Visualize generator progress
                if self.viz:
                    if(self.G_iter < 200 and self.G_iter % 10 == 0):
                        self.viz_data(save = True)
                    elif(self.G_iter >200 and self.G_iter % 200 == 0):
                        self.viz_data(save=True)
                '''
                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                if(self.gantype in ['wgan', 'wgangp', 'wganlp']):
                    # Train the generator to (roughly) minimize the approximated
                    # Wasserstein distance
                    G_loss = self.train_G_W(images)
                elif(self.gantype == 'gan'):
                    G_loss = self.train_G_vanilla(images)
                elif(self.gantype == 'nsgan'):
                    G_loss = self.train_G_ns(images)
                elif(self.gantype == 'ls'):
                    G_loss = self.train_G_ls(images)
                else:
                    print('Unknown gantype')
                    break

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()
                self.G_iter += 1
                
            

            # Save progress
            if self.viz:
                    if(self.num_epochs <= 30 and self.num_epochs % 5 == 0):
                        self.viz_data(save = True)
                    elif(self.num_epochs > 30 and self.num_epochs < 101 and self.num_epochs % 10 == 0):
                        self.viz_data(save = True)
                    elif(self.num_epochs > 100 and self.num_epochs % 20 == 0):
                        self.viz_data(save = True)
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)
            self.num_epochs += 1

            """
            # Progress logging
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1
            """
            
            
    
    def train_D_GP(self, images, LAMBDA=0.1):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise)
        G_output = self.model.G(noise)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images) # D(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon*images + (1-epsilon)*G_output
        D_interpolation = self.model.D(G_interpolation)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss
    
    def train_D_LP(self, images, LAMBDA=0.1):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[max(0,∇ D(εx + (1 − εG(z))) - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise)
        G_output = self.model.G(noise)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images) # D(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon*images + (1-epsilon)*G_output
        D_interpolation = self.model.D(G_interpolation)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        zer = torch.zeros(gradients.norm(2,dim=1).shape[0])
        grad_penalty = LAMBDA * torch.mean((torch.max(zer, gradients.norm(2, dim=1) - 1))**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss
    

    def train_D_W(self, images):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))]
        """
        # Sample from the generator
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise)
        G_output = self.model.G(noise)

        # Score real, generated images
        DX_score = self.model.D(images) # D(x), "real"
        DG_score = self.model.D(G_output) # D(G(x')), "fake"

        # Compute WGAN loss for D
        D_loss = -1 * (torch.mean(DX_score)) + torch.mean(DG_score)

        return D_loss
    
    def train_D_vanilla(self, images):
        """ Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: non-saturing loss for discriminator,
            -E[log(D(x))] - E[log(1 - D(G(z)))]
        """

        # Sample noise z, generate output G(z)
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise)
        G_output = self.model.G(noise)

        # Classify the generated and real batch images
        DX_score = self.model.D(images) # D(x)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute vanilla (original paper) D loss
        D_loss = -torch.mean(torch.log(DX_score + 1e-8)) + torch.mean(torch.log(1 - DG_score + 1e-8))
        return D_loss
    
    def train_D_ls(self, images):
        # Sample noise z, generate output G(z)
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise)
        G_output = self.model.G(noise)

        # Classify the generated and real batch images
        DX_score = self.model.D(images) # D(x)
        DG_score = self.model.D(G_output) # D(G(z))
        
        D_loss = 0.5 * (torch.mean(torch.square(DX_score-1)) + torch.mean(torch.square(DG_score)))
        return D_loss

    def train_G_W(self, images):
        """ Run 1 step of training for generator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise) # z
        G_output = self.model.G(noise) # G(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute WGAN loss for G
        G_loss = -1 * (torch.mean(DG_score))

        return G_loss
    

    def train_G_ns(self, images):
        """ Run 1 step of training for generator
        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: non-saturating loss for how well G(z) fools D,
            -E[log(D(G(z)))]
        """

        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = compute_noise(images.shape[0], self.model.pg[0], noise = self.noise) # (z)
        G_output = self.model.G(noise) # G(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute the non-saturating loss for how D did versus the generations
        # of G using sigmoid cross entropy
        G_loss = -torch.mean(torch.log(DG_score + 1e-8))

        return G_loss
    
    def train_G_vanilla(self, images):
        """ Run 1 step of training for generator
        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: minimax loss for how well G(z) fools D,
            E[log(1-D(G(z)))]
        """
        # Get noise (denoted z), classify it using G, then classify the output of G using D.
        noise = compute_noise(images.shape[0], self.model.pg[0]) # z
        G_output = self.model.G(noise) # G(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute the minimax loss for how D did versus the generations of G using sigmoid cross entropy
        G_loss = torch.mean(torch.log((1-DG_score) + 1e-8))

        return G_loss
    
    def train_G_ls(self, images):
        # Sample noise z, generate output G(z)
        noise = compute_noise(images.shape[0], self.model.pg[0])
        G_output = self.model.G(noise)

        # Classify the generated and real batch images
        DG_score = self.model.D(G_output) # D(G(z))
        
        G_loss = 0.5 * torch.mean(torch.square(DG_score-1))
        return G_loss
    
    def pretrain(self, G_init, G_optimizer):
        # Let G train for a few steps before beginning to jointly train G
        # and D because MM GANs have trouble learning very early on in training
        if G_init > 0:
            for _ in range(G_init):
                # Process a batch of images
                images = self.process_batch(self.iters)

                # Zero out gradients for G
                G_optimizer.zero_grad()

                # Pre-train G
                G_loss = self.train_G_vanilla(images)

                # Backpropagate the generator network
                G_loss.backward()
                G_optimizer.step()

            print('G pre-trained for {0} training steps.'.format(G_init))
        else:
            print('G not pre-trained -- GAN unlikely to converge.')


    def clip_D_weights(self, clip):
        for parameter in self.model.D.parameters():
            parameter.data.clamp_(-clip, clip)
            
    
    def generate_samples(self, epoch=-1, num_outputs=64):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = compute_noise(num_outputs, self.model.pg[0], noise = self.noise)

        # Transform noise to image
        images = self.model.G(noise)
        
        return images

        """
        # Reshape to square image size
        images = images.view(images.shape[0],
                             self.model.shape,
                             self.model.shape,
                             -1).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1


        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         %(epoch), nrow=grid_size)
        """

    def viz_loss(self, save = False):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.G_iter, len(self.Dlosses)),
                 self.Dlosses,
                 'b')
        print(self.num_epochs,self.G_iter, len(self.Dlosses))

        """
        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        """
        #plt.title(self.name)
        if save:
            plt.savefig('loss_%s_%d.png'%(self.gantype, self.G_iter), dpi = 150)
        plt.show()
        
        
    def viz_data(self, save = False, density=True):
        if(density == True):
            lower = (-1.3, -1.3)
            upper = (1.3, 1.3)
            nbins=300
            x,y = sep(self.generate_samlpes(num_outputs = 1000))
            k = kde.gaussian_kde([x,y])
            xi, yi = np.mgrid[lower[0]:upper[1]:nbins*1j, lower[1]:upper[1]:nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            fig, ax = plt.subplots()

            # a (1-alpha)-confidence circle for 2d gaussian with cov = a * eyes has radius sqrt(-2 a ln(alpha)), here a = 3/400
            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues)
            for i in range(len(means)):
                circle = plt.Circle((means[i][0]/10, means[i][1]/10),           # (x,y)
                np.sqrt(-3 * np.log(0.05)/200), color='black',linewidth = 2, fill=False)
                ax.add_artist(circle)
            #plt.colorbar()
            
        else:
            # 2-dim points rowwise concatenated in array
            # Set style, figure size
            plt.style.use('ggplot')
            plt.rcParams["figure.figsize"] = (8,6)

            # Plot generated points in red
            gen = self.generate_images()
            xg = [gen[i][0] for i in range(len(gen))]
            yg = [gen[i][1] for i in range(len(gen))]
            plt.plot(xg,
                     yg,
                     'ro', markersize = 3)


            # Plot real data points in green
            real = next(iter(train_iter))[0]
            xr = [real[i][0] for i in range(len(real))]
            yr = [real[i][1] for i in range(len(real))]
            plt.plot(xr,
                     yr,
                     'go', markersize = 3)

            # Add legend, title
            plt.legend(['generated', 'real'])
        
        #plt.title(self.name)
        if save:
            plt.savefig('data_%s_%d_%d.png'%(self.gantype,self.num_epochs, self.G_iter), dpi = 150)
        plt.show()
        

    def save_model(self):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), 'model_%s_%d'%(self.gantype, self.G_iter))

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        
        
        
        
        
        
        
        
        
