import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
from dataset import *
from datetime import datetime
from sampler import poisson_disk
from globals import *

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

net = SirenNet(
    dim_in=2,  # input dimension, ex. 2d coor
    dim_hidden=256,  # hidden dimension
    dim_out=1,  # output dimension, ex. rgb value
    num_layers=5,  # number of layers
    w0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

# create 2d dataset
if fix_dataset:
    train_set = Dataset.load_train_set("datasets/" + function_name)
else:
    gen = poisson_disk(r=0.05, k=100, span=[[-1, 1], [-1, 1]])
    dataset = Dataset(gen=gen)
    train_set = dataset.get_train_set()

input = torch.tensor(train_set[0])
gt = torch.tensor(train_set[1])
mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for i in range(steps):
    optimizer.zero_grad()
    eval = net(input)
    loss = mse_loss(eval, gt)
    print("Step: ", i, "; MSE Loss:", loss.item())
    loss.backward()
    optimizer.step()


print("--------------Training status: OK!--------------")
print("--------------TESTING--------------")
print("Dummy test (x, y) = (0, 0): ", net(torch.tensor([[0.0, 0.0]])))

print("---------------SAVING--------------")
current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
path = f"models/{function_name}siren_latest-{current_time}-{steps}.ckpt"
print("Save to: ", path)
torch.save(net, path)
