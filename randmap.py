from turtle import forward
import torch
import torch.nn as nn
import math
from utils import *
from relabel_voting import *

from sklearn.cluster import KMeans

class RandMap(nn.Module):
    def __init__(self, in_shape, hid_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.hid_shape = hid_shape
        self.out_shape = out_shape

        self.layer1 = nn.Linear(in_shape,hid_shape)
        self.layer2 = nn.Linear(hid_shape, out_shape)

        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(self.layer1.bias, a=-1., b=1.)
        nn.init.uniform_(self.layer2.bias, a=-1., b=1.)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x)) * math.sqrt(2/self.hid_shape)
        x = self.relu(self.layer2(x)) * math.sqrt(2/self.out_shape)
        return x

    
N = [128,512,1024]
O = [2,8,16,64]

for i in N:
    for j in O:
        data,label,classes = control()
        rm = RandMap(data.shape[1],i,j)
        out = rm(torch.tensor(data).float())

        cs = KMeans(classes)
        pred = cs.fit_predict(out.detach().numpy())
        # relabel_cluster([label,pred])
        print(f"N:{i}, O:{j} :{(relabel_cluster([label,pred])[1] == label).sum()/label.shape}")
