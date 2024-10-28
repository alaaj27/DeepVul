import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset
import random
from torch.autograd import Variable
from torch.nn import TripletMarginLoss

import numpy as np

class FC_Autoencoder(nn.Module):
    """Autoencoder"""
    def __init__(self, n_input, latent_variable_size, n_hidden=512):
        super(FC_Autoencoder, self).__init__()
        self.latent_variable_size = latent_variable_size
        self.n_input = n_input
        self.n_hidden = n_hidden

        
        self.encoder = nn.Sequential(
            nn.Linear(self.n_input, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
        )
        

        self.fc1 = nn.Linear(n_hidden, latent_variable_size)
        self.fc2 = nn.Linear(n_hidden, latent_variable_size)


        self.decoder = nn.Sequential(
            nn.Linear(latent_variable_size, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_input),
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_space = self.reparametrize(mu, logvar)
        res = self.decode(latent_space)
        return res, latent_space, mu, logvar
    

    def encode(self, x):
        h = self.encoder(x)       
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res
