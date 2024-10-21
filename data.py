import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Define the data, and any transformations which happen to it in this file
"""

def get_data(batch_size, transform_indices):
    # load in the data
    # a series of transforms to augment our dataset before training
    # normalization params lovingly stolen from https://github.com/poojahira/gtsrb-pytorch/blob/master/data.py
    normalization_params = transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

    training_transforms = [
        # basic transpose, no effect
        transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalization_params
            ]
        ),

        # jitter image brightness
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=10),
            transforms.ToTensor(),
            normalization_params
        ]),    

        # random rotation
        transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.RandomRotation(3),
            transforms.ToTensor(),
            normalization_params
        ]),

        # elastic "waterfall" effect
        transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ToTensor(),
            normalization_params
        ]),

        # gaussian blur
        transforms.Compose([
            transforms.Resize((32, 32)),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            transforms.ToTensor(),
            normalization_params
        ])
    ]

    test_transform = transforms.Compose( [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
    ])

    BATCH_SIZE = 4

    # concatenate altered data together
    training_transforms = [training_transforms[i] for i in transform_indices]
    print(training_transforms)
    trainset = torch.utils.data.ConcatDataset(
        [
        torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=curr_transform) for curr_transform in training_transforms
        ]
    )
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader, testloader

if __name__=="__main__":
    trainloader, testloader = get_data(1)
    print(len(testloader))
    print(len(trainloader))