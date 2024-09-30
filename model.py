import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((500,500)) #TODO reconsider size transform, and transform in general
    ]
)

batch_size = 4

trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

for data, target in trainloader:
    print(target)