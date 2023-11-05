import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision.models import resnet18

import math
from utils.config.configurations import cfg


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

class ANOSHIFT_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,500,150,50,10,1]
        super(ANOSHIFT_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)        


class CICIDS2017_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,150,50,1]
        super(CICIDS2017_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class CICIDS2018_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,200,50,10,1]
        super(CICIDS2018_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)

class UNSWNB15_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,150,50,1]
        super(UNSWNB15_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)        

class CIDDS_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,500,250,50,1]
        super(CIDDS_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)   



class KDDCUP99_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,150,50,1]
        super(KDDCUP99_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x) 


class custom_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(custom_LSTM, self).__init__()
        
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.output_size = output_size
        

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.fc = nn.Linear(hidden_size, self.output_size) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(cfg.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(cfg.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc(out) 
        
        return out



class CIFAR10_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,500,100,50,1]
        super(CIFAR10_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class MNIST_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,150,50,10,1]
        super(MNIST_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)  


class SVHN_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,150,50,10,1]
        super(SVHN_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)          


class RESNET_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,50,1]
        super(RESNET_FC, self).__init__()
         
        model = resnet18(weights=None)       
        model.fc = nn.Sigmoid()
        self.net = model

        
    def forward(self, x):
        return self.net(x)       


class Weights_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,2,inputsize]
        super(Weights_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)    



class CLEAR10_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,150,100,50,1]
        super(CLEAR10_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)    


class SMAP_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,150,50,1]
        super(SMAP_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)   





class ICARL_FC(nn.Module):
    def __init__(self,inputsize=512):
        sizes = [inputsize,100,50,1]
        super(ICARL_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)                              


              
