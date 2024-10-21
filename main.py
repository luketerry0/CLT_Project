import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import get_data
from model import get_model

"""
Classifier training and verification
"""

def main():
    # pull in data and model from other files
    net, criterion, optimizer = get_model()
    trainloader, testloader = get_data(batch_size=50)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = train(net, criterion, optimizer, trainloader, device, testloader)
    # evaluate(testloader, net, device)

def train(net, criterion, optimizer, trainloader, device, testloader):
    # use CUDA, if available
    net.to(device)
    keep_training = True
    acc = 0
    last_acc = 0
    epoch = 0

    while keep_training:  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 100 batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0
        
        # evaluate training accuracy and decide whether to stop
        acc = evaluate(testloader, net, device)
        if (acc < last_acc):
            keep_training = False
        epoch += 1
        last_acc = acc

    print('Finished Training')
    return net

def evaluate(testloader, net, device):
    correct = 0
    total = 0
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return (100 * correct / total)

if __name__=="__main__":
    main()