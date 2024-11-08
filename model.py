import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Defining the structure of the model used in this file
"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.ReLU()
        )  


    def forward(self, x):
        return self.layers(x)

def get_model(learning_rate):
    # define the criteria for training the neural net
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return net, criterion, optimizer