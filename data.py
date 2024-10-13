import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Define the data, and any transformations which happen to it in this file
"""

def get_data(batch_size):
    # load in the data
    train_transform = transforms.Compose(
        # transform lovingly stolen from https://github.com/poojahira/gtsrb-pytorch/blob/master/data.py
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ]
    )

    test_transform = transforms.Compose( [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
    ])

    BATCH_SIZE = 4

    trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader, testloader

if __name__=="__main__":
    trainloader, testloader = get_data(1)
    print(len(testloader))
    print(len(trainloader))