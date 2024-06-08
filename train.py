import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
from dataset import *

"""
1. Train the model using train dataset
2. Do the validation process using test dataset

NOTE on how to calculate d(NN) / d(input):
    # test grad
    input_ = torch.tensor([[0.5, 0.5]], requires_grad=True)
    output_ = net(input_)
    output_.backward()
    print(output_)
    print(input_.grad)
"""

steps = 100

net = SirenNet(
    dim_in=2,  # input dimension, ex. 2d coor
    dim_hidden=256,  # hidden dimension
    dim_out=1,  # output dimension, ex. rgb value
    num_layers=5,  # number of layers
    w0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

# prepare the 2D function data
dataset = Dataset(10000)
train_set = dataset.get_train_set()
test_set = dataset.get_test_set()


mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for i in range(steps):
    optimizer.zero_grad()
    eval = net(train_set[0])
    loss = mse_loss(eval, train_set[1])
    print("Step: ", i, "; MSE Loss:", loss.item())
    loss.backward()
    optimizer.step()

print("--------------Training status: OK!--------------")
print("--------------Validation Result--------------")
test_output = net(test_set[0])
print(mse_loss(test_output, test_set[1]).item())
print("Dummy test (x, y) = (0, 0): ", net(torch.tensor([[0.0, 0.0]])))

print("---------------Saving--------------")
torch.save(net, f"siren_latest-{steps}.ckpt")
