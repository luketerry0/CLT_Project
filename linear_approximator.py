# this file implements a linear approximator  intended to make a neural net more explainable
# as described in https://arxiv.org/abs/1602.04938
# LIME (Locally Interpretable Model-Agnostic Explaination)

from data import get_data
from model import get_model
import torch
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np
from numpy.random import choice

# load the model
model, criterion, optimizer = get_model(0.0001)
model.load_state_dict(torch.load('./model_weights.pth', weights_only=True))
model.eval()


# fetch all images
BATCH_SIZE = 60000
trainloader, testloader = get_data(BATCH_SIZE)
inputs, outputs = next(iter(trainloader))
inputs = inputs.resize(60000,784)


# local weighting for squared distance function.
# width is a parameter adjusting the degree of locality to consider
def pi(x, z, width=15):
    # get the (l2 euclidean) distance between the two images
    distance = torch.norm(x - z, p=2)
    term = (distance**2)/(width**2)
    return torch.exp(-1*term)

# locally weighted squared loss
# z is our actual representation, z_prime the interpretable representation,
#  and g the simple model to be applied to z_prime
# here for completeness, this process is approximmated by LIME below....
def L(z, z_prime, g, p=pi):
    return pi(z, z_prime)*(model(z) - model(z_prime))**2

# algorithm  1 from the paper, the important one
# approximates a locally linear model using K-lasso
# x is the instance in question, 
# N is the number of samples to take around S, 
# K is the length of the simple explaination, 
# and the model is our real classifer
def LIME(x, N, K, model=model):
    # get all the data
    loader, _ = get_data(60000)
    inputs, outputs = next(iter(loader))
    inputs = inputs.resize(60000,784)

    # take weighted sample based on pi function for each data point
    result = np.array(torch.stack([pi(x, row) for row in inputs])).astype('float64')
    result /= sum(result)
    chosen_inputs = np.random.choice(60000, N, p=result)

    inputs = inputs[chosen_inputs].detach()
    outputs = outputs[chosen_inputs].detach()

    # convert these samples into interpretable versions using K-Lasso
    # select K features using LASSO
    lasso = Lasso(alpha=0.05)
    lasso.fit(inputs, outputs)
    selected_features = SelectFromModel(lasso, max_features=K)
    support = selected_features.get_support()

    # visualize selected features
    features = np.array(support).astype(int)
    feature_viz = np.resize(features, (28, 28)).astype(float)
    K = sum(features)
    # print(sum(features))
    # plt.imshow(feature_viz, cmap="gray")
    # plt.show()

    # make simplified model
    simplified_data = inputs[:,support]
    simple_model = LinearRegression()
    real_model_outputs = model(inputs).detach()
    simple_model.fit(simplified_data, real_model_outputs)

    return feature_viz, simple_model, K


pic_idx = 10
ip = inputs[pic_idx]
viz, simple_model, K = LIME(ip, 20000, 100, model)
print(K)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(ip.resize(28, 28), cmap='gray')
axes[1].imshow(viz, cmap='gray')
plt.show()
        



